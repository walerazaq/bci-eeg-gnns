# Leave One Subject Out (LOSO) Training and Validation Function

def train(model, dataset, bs, lrate, wd, device):
    model = model.to(device)

    loss_function = LabelSmoothingCrossEntropy()
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=wd)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-5)

    best_models = []
    best_metrics = []
    fold = 0
    
    subjects = [1,2,3,4,5,6,7,8,9,10]
    sub_count = len(subjects)
    
    for i in subjects:
        model._reset_parameters()
        
        val_dataset = []
        train_dataset = []
        
        for j in range(len(dataset)):
            if dataset[j].sub == i:
                val_dataset.append(dataset[j])
            else:
                train_dataset.append(dataset[j])

        train_loader = DataLoader(
            train_dataset,
            batch_size=bs,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=bs,
            shuffle=True
        )

        best_metric = -1
        best_metric_epoch = -1
        best_val_loss = 1000
        best_model = None
        epochs = 1000

        print('-' * 30)
        print('Training ... ')
        early_stop = 30
        es_counter = 0
        
        fold += 1
        print(f"Fold {fold}/{sub_count}")

        for epoch in range(epochs):

            print("-" * 10)
            print(f"epoch {epoch + 1}/{epochs}")
            model.train()
            epoch_train_loss = 0

            for i, data in enumerate(tqdm(train_loader)):
                batch = data.batch.to(device)
                x = data.x.to(device)
                y = data.y.to(device)
                u = data.adj.to(device)
                optimizer.zero_grad()

                out = model(x, u, batch)

                step_loss = loss_function(out, y)
                step_loss.backward(retain_graph=True)
                optimizer.step()
                epoch_train_loss += step_loss.item()

            epoch_train_loss = epoch_train_loss / (i + 1)
            lr_scheduler.step()
            val_loss, val_acc = validate_model(model, val_loader, device)
            print(f"epoch {epoch + 1} train loss: {epoch_train_loss:.4f}")

            if val_loss < best_val_loss:
                best_metric = val_acc
                best_val_loss = val_loss
                best_metric_epoch = epoch + 1
                best_model = deepcopy(model)
                print("saved new best metric model")
                es_counter = 0
            else:
                es_counter += 1

            if es_counter > early_stop:
                print('No loss improvement.')
                break

            print(
                "current epoch: {} current val loss {:.4f} current accuracy: {:.4f}  best accuracy: {:.4f} at loss {:.4f} at epoch {}".format(
                    epoch + 1, val_loss, val_acc, best_metric, best_val_loss, best_metric_epoch))

        print(f"train completed, best_val_loss: {best_val_loss:.4f} at epoch: {best_metric_epoch}")

        best_models.append(best_model)
        best_metrics.append((best_metric, best_val_loss))

    return best_models, best_metrics

def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0
    loss_func = nn.CrossEntropyLoss()

    labels = []
    preds = []
    for i, data in enumerate(val_loader):
            batch = data.batch.to(device)
            x = data.x.to(device)
            label = data.y.to(device)
            u = data.adj.to(device)

            out = model(x,u,batch)

            step_loss = loss_func(out, label)
            val_loss += step_loss.detach().item()
            preds.append(out.argmax(dim=1).detach().cpu().numpy())
            labels.append(label.cpu().numpy())
    preds = np.concatenate(preds).ravel()
    labels =  np.concatenate(labels).ravel()
    acc = balanced_accuracy_score(labels, preds)
    loss = val_loss/(i+1)

    return loss, acc
