from flcore.models.cnn.fedavg_cnn import FedAvgCNN




def get_model(args, ):
    if args.model == 'CNN':
        model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
    return model
