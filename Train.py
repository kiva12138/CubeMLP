import os
import random
import logging

import numpy as np
import torch
from scipy.stats.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from torch.utils.tensorboard import SummaryWriter

from Config import CUDA
from DataLoaderLocal import mosi_r2c_7, pom_r2c_7, r2c_2, r2c_7
from DataLoaderUniversal import get_data_loader
from Model import Model
from Parameters import parse_args
from Utils import SAM, SIMSE, get_mask_from_sequence, rmse, set_logger, topk_, calc_metrics, calc_metrics_pom, to_gpu

from transformers import BertTokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def set_random_seed(opt):
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_cuda(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA
    torch.cuda.set_device("cuda:" + opt.cuda)


def prepare_ckpt_log(opt):
    os.makedirs(os.path.join(opt.ckpt_path, opt.task_name), exist_ok=True)
    os.makedirs(os.path.join(opt.log_path, opt.task_name, "predictions"), exist_ok=True)
    set_logger(os.path.join(opt.log_path, opt.task_name, "log.log"))
    writer = SummaryWriter(os.path.join(opt.log_path, opt.task_name))
    best_model_name_val = os.path.join(opt.ckpt_path, opt.task_name, "best_model_val.pth.tar")
    best_model_name_test = os.path.join(opt.ckpt_path, opt.task_name, "best_model_test.pth.tar")
    ckpt_model_name = os.path.join(opt.ckpt_path, opt.task_name, "latest_model.pth.tar")
    return writer, best_model_name_val, best_model_name_test, ckpt_model_name


def other_model_operations(model, opt):    
    for name, param in model.named_parameters():
        if opt.bert_freeze=='part' and "bertmodel.encoder.layer" in name:
            layer_num = int(name.split("encoder.layer.")[-1].split(".")[0])
            if layer_num <= (8):
                param.requires_grad = False
        elif opt.bert_freeze=='all' and "bert" in name:
            param.requires_grad = False
        
        if 'weight_hh' in name:
            torch.nn.init.orthogonal_(param)
        if opt.print_param:
            print('\t' + name, param.requires_grad)


def get_optimizer(opt, model):
    if opt.bert_lr_rate <= 0:
        params = filter(lambda p: p.requires_grad, model.parameters())
    else:
        def get_berts_params(model):
            results = []
            for p in model.named_parameters():
                if 'bert' in p[0] and p[1].requires_grad:
                    results.append(p[1])
            return results
        def get_none_berts_params(model):
            results = []
            for p in model.named_parameters():
                if 'bert' not in p[0] and p[1].requires_grad:
                    results.append(p[1])
            return results
        params = [
            {'params': get_berts_params(model), 'lr': float(opt.learning_rate) * opt.bert_lr_rate},
            {'params': get_none_berts_params(model), 'lr': float(opt.learning_rate)},
        ]
    if opt.optm == "Adam":
        optimizer = torch.optim.Adam(params, lr=float(opt.learning_rate), weight_decay=opt.weight_decay)
    elif opt.optm == "SGD":
        optimizer = torch.optim.SGD(params, lr=float(opt.learning_rate), weight_decay=opt.weight_decay, momentum=0.9 )
    elif opt.optm == "SAM":
        optimizer = SAM(params, torch.optim.Adam, lr=float(opt.learning_rate), weight_decay=opt.weight_decay,)
    else:
        raise NotImplementedError

    if opt.lr_decrease == 'step':
        opt.lr_decrease_iter = int(opt.lr_decrease_iter)
        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_decrease_iter, opt.lr_decrease_rate)
    elif opt.lr_decrease == 'multi_step':
        opt.lr_decrease_iter = list((map(int, opt.lr_decrease_iter.split('-'))))
        lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.lr_decrease_iter, opt.lr_decrease_rate)
    elif opt.lr_decrease == 'exp':
        lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, opt.lr_decrease_rate)    
    elif opt.lr_decrease == 'plateau':
        mode = 'min' # if opt.task == 'regression' else 'max'
        lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, patience=int(opt.lr_decrease_iter), factor=opt.lr_decrease_rate,)
    else:
        raise NotImplementedError
    return optimizer, lr_schedule


def get_loss(opt):
    if opt.loss == 'RMSE':
        loss_func = rmse
    elif opt.loss == 'MAE':
        loss_func = torch.nn.L1Loss()
    elif opt.loss == 'MSE':
        loss_func = torch.nn.MSELoss(reduction='mean')
    elif opt.loss == 'SIMSE':
        loss_func = SIMSE()
    else :
        raise NotImplementedError
        
    return [loss_func]


def train(train_loader, model, optimizer, loss_func, opt):
    model.train()
    running_loss, predictions_corr, targets_corr = 0.0, [], []

    for _, datas in enumerate(train_loader):
        t_data, a_data, v_data = datas[0], datas[1].cuda().float(), datas[2].cuda().float()
        labels = get_labels_from_datas(datas, opt) # Get multiple labels
        targets = get_loss_label_from_labels(labels, opt).cuda() # Get target from labels

        outputs = get_outputs_from_datas(model, t_data, a_data, v_data, opt) # Get multiple outputs
        loss = get_loss_from_loss_func(outputs, targets, loss_func, opt)  # Get loss

        optimizer.zero_grad()
        loss.backward()
        if opt.gradient_clip > 0:
            torch.nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], opt.gradient_clip)
        optimizer.step()
        running_loss += loss.item()

        with torch.no_grad():
            predictions_corr += outputs[0].cpu().numpy().tolist()
            targets_corr += targets.cpu().numpy().tolist()

    predictions_corr, targets_corr = np.array(predictions_corr), np.array(targets_corr)
    train_score = get_score_from_result(predictions_corr, targets_corr, opt) # return is a dict

    return running_loss/len(train_loader), train_score


def evaluate(val_loader, model, loss_func, opt):
    model.eval()
    running_loss, predictions_corr, targets_corr = 0.0, [], []
    with torch.no_grad():
        for _, datas in enumerate(val_loader):
            t_data, a_data, v_data = datas[0], datas[1].cuda().float(), datas[2].cuda().float()
            labels = get_labels_from_datas(datas, opt) # Get multiple labels
            targets = get_loss_label_from_labels(labels, opt).cuda() # Get target from labels

            outputs = get_outputs_from_datas(model, t_data, a_data, v_data, opt) # Get multiple outputs
            loss = get_loss_from_loss_func(outputs, targets, loss_func, opt)  # Get loss
            running_loss += loss.item()

            predictions_corr += outputs[0].cpu().numpy().tolist()
            targets_corr += targets.cpu().numpy().tolist()

    predictions_corr, targets_corr = np.array(predictions_corr), np.array(targets_corr)
    valid_score = get_score_from_result(predictions_corr, targets_corr, opt) # return is a dict

    return running_loss/len(val_loader), valid_score, predictions_corr, targets_corr


def main():
    opt = parse_args()
    
    set_cuda(opt)
    set_random_seed(opt)
    writer, best_model_name_val, best_model_name_test, _ = prepare_ckpt_log(opt)
    
    logging.log(msg=str(opt), level=logging.DEBUG)

    logging.log(msg="Making dataset, model, loss and optimizer...", level=logging.DEBUG)
    train_loader, valid_loader, test_loader = get_data_loader(opt)
    model = Model(opt)
    other_model_operations(model, opt)
    optimizer, lr_schedule = get_optimizer(opt, model)
    loss_func = get_loss(opt)
    if opt.parallel:
        logging.log(msg="Model paralleling...", level=logging.DEBUG)
        model = torch.nn.DataParallel(model, device_ids=list(map(int, CUDA.split(','))))
    model = model.cuda()

    logging.log(msg="Start training...", level=logging.DEBUG)
    best_score_val, best_score_test, best_score_test_in_valid  = None, None, None
    best_val_predictions, best_test_predictions, best_test_predictions_in_valid = None, None, None
    for epoch in range(opt.epochs_num):
        # Do Train and Evaluate
        train_loss, train_score = train(train_loader, model, optimizer, loss_func, opt)
        val_loss, val_score, val_predictions, val_targets = evaluate(valid_loader, model, loss_func, opt)
        test_loss, test_score, test_predictions, test_targets = evaluate(test_loader, model, loss_func, opt)
        if opt.lr_decrease == 'plateau':
            lr_schedule.step(test_loss)
        else:
            lr_schedule.step()

        # Updata metrics, results and features
        if current_result_better(best_score_val, val_score, opt):
            best_model_state_val = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
            }
            best_score_val, best_val_predictions = val_score, val_predictions
            best_score_test_in_valid, best_test_predictions_in_valid = test_score, test_predictions
            logging.log(msg='Better Valid score found...', level=logging.DEBUG)
            calc_metrics(val_targets, val_predictions)
            logging.log(msg='Test in Better Valid score found...', level=logging.DEBUG)
            calc_metrics(test_targets, test_predictions)

        # Log the epoch result
        msg = build_message(epoch, train_loss, train_score, val_loss, val_score, test_loss, test_score)
        logging.log(msg=msg, level=logging.DEBUG)
        log_tf_board(writer, epoch, train_loss, train_score, val_loss, val_score, test_loss, test_score, lr_schedule)

    # Log the best
    logging.log(msg=build_single_message(best_score_val, 'Best Valid Score \t\t'), level=logging.DEBUG)
    logging.log(msg=build_single_message(best_score_test_in_valid, 'Test Score at Best Valid \t'), level=logging.DEBUG)
    writer.close()

    # Save predictions
    np.save(os.path.join(opt.log_path, opt.task_name, "predictions", "val.npy"), best_val_predictions)
    np.save(os.path.join(opt.log_path, opt.task_name, "predictions", "test_for_valid.npy"), best_test_predictions_in_valid)

    # Save model
    torch.save(best_model_state_val, best_model_name_val)


def current_result_better(best_score, current_score, opt):
    if best_score is None:
        return True
    if opt.task == 'classification':
        return current_score[str(opt.num_class)+'-class_acc'] > best_score[str(opt.num_class)+'-class_acc']
    elif opt.task == 'regression':
        return current_score['mae'] < best_score['mae']
    else:
        raise NotImplementedError


def get_labels_from_datas(datas, opt):
    if 'SDK' in opt.dataset:
        return datas[3:-1]
    else:
        return datas[3:]


def get_loss_label_from_labels(labels, opt):
    if opt.dataset in ['mosi_SDK', 'mosei_SDK', 'mosi_20', 'mosi_50', 'mosei_20', 'mosei_50']:
        if opt.task == 'regression':
            labels = labels[0]
        elif opt.task == 'classification' and opt.num_class==2:
            labels = labels[1]
        elif opt.task == 'classification' and opt.num_class==7:
            labels = labels[2]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return labels


def get_outputs_from_datas(model, t_data, a_data, v_data, opt):
    if a_data.shape[1] > opt.time_len:
        a_data = a_data[:, :opt.time_len, :]
        v_data = v_data[:, :opt.time_len, :]
        t_data = [sample[:opt.time_len] for sample in t_data]
    sentences = [" ".join(sample) for sample in t_data]
    bert_details = bert_tokenizer.batch_encode_plus(sentences, add_special_tokens=True, padding=True)
    # print(encoded_bert_sent)
    bert_sentences = to_gpu(torch.LongTensor(bert_details["input_ids"]))
    bert_sentence_types = to_gpu(torch.LongTensor(bert_details["token_type_ids"]))
    bert_sentence_att_mask = to_gpu(torch.LongTensor(bert_details["attention_mask"]))
    outputs = model(bert_sentences, bert_sentence_types, bert_sentence_att_mask, a_data, v_data, return_features=True)

    return outputs


def get_loss_from_loss_func(outputs, labels, loss_func, opt):
    # Get predictions
    # predictions, T_F_, A_F_, V_F_ = outputs[0], outputs[1], outputs[2] ,outputs[3]
    predictions = outputs[0]
    task_loss = loss_func[0]

    # Get loss from predictions
    if opt.loss in ['RMSE', 'MAE', 'MSE', 'SIMSE']:
        loss = task_loss(predictions.reshape(-1, ), labels.reshape(-1, ))
    else:
        raise NotImplementedError

    # Get loss from features

    return loss


def get_score_from_result(predictions_corr, labels_corr, opt):
    if opt.task == 'classification':
        if opt.num_class == 1:
            predictions_corr = np.int64(predictions_corr.reshape(-1,) > 0)
        else:
            _, predictions_corr = topk_(predictions_corr, 1, 1)
        predictions_corr, labels_corr = predictions_corr.reshape(-1,), labels_corr.reshape(-1,)
        acc = accuracy_score(labels_corr, predictions_corr)
        f1 = f1_score(labels_corr, predictions_corr, average='weighted')
        return {
            str(opt.num_class)+'-cls_acc': acc,
            str(opt.num_class)+'-f1': f1
        }
    elif opt.task == 'regression':
        predictions_corr, labels_corr = predictions_corr.reshape(-1,), labels_corr.reshape(-1,)
        mae = mean_absolute_error(labels_corr, predictions_corr)
        corr, _ = pearsonr(predictions_corr, labels_corr )

        if opt.dataset in ['mosi_SDK', 'mosei_SDK', 'mosi_20', 'mosi_50', 'mosei_20', 'mosei_50']:
            if 'mosi' in opt.dataset:
                predictions_corr_7 = [mosi_r2c_7(p) for p in predictions_corr]
                labels_corr_7 = [mosi_r2c_7(p) for p in labels_corr]
            else:
                predictions_corr_7 = [r2c_7(p) for p in predictions_corr]
                labels_corr_7 = [r2c_7(p) for p in labels_corr]

            predictions_corr_2 = [r2c_2(p) for p in predictions_corr]
            labels_corr_2 = [r2c_2(p) for p in labels_corr]
            acc_7 = accuracy_score(labels_corr_7, predictions_corr_7)
            acc_2 = accuracy_score(labels_corr_2, predictions_corr_2)
            f1_2 = f1_score(labels_corr_2, predictions_corr_2, average='weighted')
            f1_7 = f1_score(labels_corr_7, predictions_corr_7, average='weighted')

            return {
                'mae': mae,
                'corr': corr,
                '7-cls_acc': acc_7,
                '2-cls_acc': acc_2,
                '7-f1': f1_7,
                '2-f1': f1_2,
            }
        elif opt.dataset in ['pom_SDK', 'pom']:
            predictions_corr_7 = [pom_r2c_7(p) for p in predictions_corr]
            labels_corr_7 = [pom_r2c_7(p) for p in labels_corr]
            acc_7 = accuracy_score(labels_corr_7, predictions_corr_7)
            f1_7 = f1_score(labels_corr_7, predictions_corr_7, average='weighted')
            return {
                'mae': mae,
                'corr': corr,
                '7-cls_acc': acc_7,
                '7-f1': f1_7,
            }
        elif opt.dataset in ['mmmo', 'mmmov2']:
            predictions_corr_2 = [int(p>=3.5) for p in predictions_corr]
            labels_corr_2 = [int(p>=3.5) for p in labels_corr]
            acc_2 = accuracy_score(labels_corr_2, predictions_corr_2)
            f1_2 = f1_score(labels_corr_2, predictions_corr_2, average='weighted')

            return {
                'mae': mae,
                'corr': corr,
                '2-cls_acc': acc_2,
                '2-f1': f1_2,
            }
        else:
            raise NotImplementedError
    else :
        raise NotImplementedError


def build_message(epoch, train_loss, train_score, val_loss, val_score, test_loss, test_score):
    msg = "Epoch:[{:3.0f}]".format(epoch + 1)
    
    msg += " ||"
    msg += " TrainLoss:[{0:.3f}]".format(train_loss)
    for key in train_score.keys():
        msg += " Train_"+key+":[{0:6.3f}]".format(train_score[key])

    msg += " ||"
    msg += " ValLoss:[{0:.3f}]".format(val_loss)
    for key in val_score.keys():
        msg += " Val_"+key+":[{0:6.3f}]".format(val_score[key])

    return msg


def build_single_message(best_score, mode):
    msg = mode
    for key in best_score.keys():
        msg += " "+key+":[{0:6.3f}]".format(best_score[key])
    return msg


def log_tf_board(writer, epoch, train_loss, train_score, val_loss, val_score, test_loss, test_score, lr_schedule):
    writer.add_scalar('Train/Epoch/Loss', train_loss, epoch)
    for key in train_score.keys():
        writer.add_scalar('Train/Epoch/'+key, train_score[key], epoch)
    writer.add_scalar('Valid/Epoch/Loss', val_loss, epoch)
    for key in val_score.keys():
        writer.add_scalar('Valid/Epoch/'+key, val_score[key], epoch)
    try:
        writer.add_scalar('Lr',  lr_schedule.get_last_lr()[-1], epoch)
    except:
        pass


if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()
    main()
