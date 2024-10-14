import my_nlp_library as nlp
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

def train_binary_model(model, dataset, n_epochs=100, batch_size=64, lr=1e-3):
    if torch.cuda.is_available():
        use_cuda = True
    else:
        use_cuda = False

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # lr is the learning rate - this is our alpha
    loss_fn = nn.BCEWithLogitsLoss() # Binary Cross Entropy from Logits
    # And now, a loop that is equal for everyone:
    losses = []
    if use_cuda == True:
        model = model.cuda()

    for epoch in tqdm(range(n_epochs)):
        epoch_loss = 0
        for batch in dataloader:
            if use_cuda == True:
                batch = [item.cuda() for item in batch]
            X_train_vect, y_train_vect = batch
            optimizer.zero_grad()
            output = model(X_train_vect)
            loss = loss_fn(output, y_train_vect.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        losses.append(epoch_loss / len(dataloader))
    if use_cuda == True:
        model = model.cpu()
    return model, losses

def test_binary_model(model, dataset, batch_size=64):
    dataloader_test = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        acc = 0
        n_tests = 0
        for X_test_vect, y_test_vect in dataloader_test:
            output = model(X_test_vect)
            output_probs = torch.sigmoid(output)
            predictions = (output_probs > 0.5).int()

            # get batch accuracy
            accuracy = accuracy_score(y_test_vect, predictions)
            acc += accuracy * len(y_test_vect)
            n_tests += len(y_test_vect)
    # Calculate accuracy
    accuracy = acc / n_tests
    return accuracy