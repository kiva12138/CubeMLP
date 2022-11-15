from DataLoaderCMUSDK import *

def get_data_loader(opt):
    dataset = opt.dataset
    text, audio, video=opt.text, opt.audio, opt.video # Only for CMUSDK dataset
    normalize = opt.normalize
    persistent_workers=opt.persistent_workers
    batch_size, num_workers, pin_memory, drop_last =opt.batch_size, opt.num_workers, opt.pin_memory, opt.drop_last

    assert dataset in ['mosi_SDK', 'mosei_SDK']

    if 'mosi' in dataset:
        dataset_train = CMUSDKDataset(mode='train', dataset='mosi', text=text, audio=audio, video=video, normalize=normalize, )
        dataset_valid = CMUSDKDataset(mode='valid', dataset='mosi', text=text, audio=audio, video=video, normalize=normalize, )
        dataset_test = CMUSDKDataset(mode='test', dataset='mosi', text=text, audio=audio, video=video, normalize=normalize, )
        data_loader_train = DataLoader(dataset_train, batch_size, collate_fn=multi_collate_mosei_mosi, shuffle=True, 
            persistent_workers=persistent_workers, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
        data_loader_valid = DataLoader(dataset_valid, batch_size, collate_fn=multi_collate_mosei_mosi, shuffle=False, 
            persistent_workers=persistent_workers, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
        data_loader_test = DataLoader(dataset_test, batch_size, collate_fn=multi_collate_mosei_mosi, shuffle=False, 
            persistent_workers=persistent_workers, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
        return data_loader_train, data_loader_valid, data_loader_test
    
    if 'mosei' in dataset:
        dataset_train = CMUSDKDataset(mode='train', dataset='mosei', text=text, audio=audio, video=video, normalize=normalize, )
        dataset_valid = CMUSDKDataset(mode='valid', dataset='mosei', text=text, audio=audio, video=video, normalize=normalize, )
        dataset_test = CMUSDKDataset(mode='test', dataset='mosei', text=text, audio=audio, video=video, normalize=normalize, )
        data_loader_train = DataLoader(dataset_train, batch_size, collate_fn=multi_collate_mosei_mosi, shuffle=True, 
            persistent_workers=persistent_workers, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
        data_loader_valid = DataLoader(dataset_valid, batch_size, collate_fn=multi_collate_mosei_mosi, shuffle=False, 
            persistent_workers=persistent_workers, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
        data_loader_test = DataLoader(dataset_test, batch_size, collate_fn=multi_collate_mosei_mosi, shuffle=False, 
            persistent_workers=persistent_workers, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
        return data_loader_train, data_loader_valid, data_loader_test

    raise NotImplementedError

