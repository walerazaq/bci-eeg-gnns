def trainEEG(config, train_dataset, val_dataset, device):
    model = GCN(8, config["f1"], 4, readout='meanmax')
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = config["lr"], weight_decay = config["wd"])
    
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
     
    # Dataloaders    
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=1
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=1
    )
    
    epochs = 2000

    for epoch in range(epochs):  # loop over the dataset multiple times
        epoch_train_loss = 0.0
        epoch_steps = 0
        
        for i, data in enumerate(tqdm(train_loader)):
            batch = data.batch.to(device)
            x = data.x.to(device)
            y = data.y.to(device)
            u = data.adj.to(device)
            optimizer.zero_grad()

            out = model(x,u,batch)

            step_loss = loss_function(out, y)
            step_loss.backward(retain_graph=True)
            optimizer.step()
            epoch_train_loss += step_loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, epoch_train_loss / epoch_steps)
                )
                epoch_train_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        loss_func = nn.CrossEntropyLoss()
        
        labels = []
        preds = []
        for i, data in enumerate(val_loader):
            with torch.no_grad():
                batch = data.batch.to(device)
                x = data.x.to(device)
                label = data.y.to(device)
                u = data.adj.to(device)

                out = model(x,u,batch)
                
                step_loss = loss_func(out, label)
                val_loss += step_loss.detach().item()
                val_steps += 1
                preds.append(out.argmax(dim=1).detach().cpu().numpy())
                labels.append(label.cpu().numpy())
                
        preds = np.concatenate(preds).ravel()
        labels =  np.concatenate(labels).ravel()
        acc = balanced_accuracy_score(labels, preds)
        
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                {"loss": (val_loss / val_steps), "accuracy": acc},
                checkpoint=checkpoint,
            )
        
    print("Finished Training")

def test_model(best_result, test_loader, device):
         
    best_trained_model = GCN(8, best_result.config["f1"], 4, readout='meanmax')

    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)
    
    best_trained_model.eval()
    labels = []
    preds = []
    for i, data in enumerate(test_loader):
            batch = data.batch.to(device)
            x = data.x.to(device)
            label = data.y.to(device)
            u = data.adj.to(device)

            out = best_trained_model(x,u,batch)
            preds.append(out.argmax(dim=1).detach().cpu().numpy())
            labels.append(label.cpu().numpy())
    preds = np.concatenate(preds).ravel()
    labels =  np.concatenate(labels).ravel()

    accuracy = balanced_accuracy_score(labels, preds)

    return accuracy

def mainTrain(num_samples=10, max_num_epochs=50, gpus_per_trial=2):
    config = {
        "f1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-5, 1e-1),
        "wd": tune.loguniform(1e-5, 1e-1),
        #"chebFilterSize": tune.choice([1, 2, 4, 8, 16]),
        #"num_heads": tune.choice([1, 2, 4, 8, 16]),
        "batch_size": tune.choice([2, 4, 8, 16]),
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=3,
        reduction_factor=2,
    )
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(trainEEG, train_dataset=train_dataset, val_dataset=val_dataset, device=device),
            resources={"cpu": 4, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    results = tuner.fit()

    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

    test_acc = test_model(best_result, test_loader, device)
    print("Best trial test set accuracy: {}".format(test_acc))
    
    return best_result