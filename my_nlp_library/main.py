import my_nlp_library as nlp
import os
from pathlib import Path
from rich.console import Console
import typer

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command('imdb')
def test_with_imdb(model : str = "baseline",
                   sample_size : int = 1000,
                   embedding_dim : int = 50,
                   hidden_dim : int = 50,
                   n_hidden_layers_mlp : int = 1,
                   n_layers_rnn : int = 1,
                   n_epochs : int = 100,
                   ):
    """
    Test baseline model with imdb
    """
    if 'glove' in model:
        use_glove = True
    else:
        use_glove = False
    dataset_train, dataset_test, tokenizer, glove_vectors = nlp.get_imdb_dataset(sample_size=sample_size, glove=use_glove)
    vocab_size = tokenizer.vocab_size

    if model == "baseline":
        model = nlp.MyClassifier(vocab_size=vocab_size, embedding_dim=embedding_dim, output_dim=1)
    elif model == "mlp":
        model = nlp.MyNetwork(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers_mlp, output_dim=1)
    elif model == "resmlp":
        model = nlp.MyResidualNetwork(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers_mlp, output_dim=1)
    elif model == "glovemlp":
        model = nlp.MyMLPResidualNetworkWithGloveEmbeddings(hidden_dim=hidden_dim, glove_vectors=glove_vectors, n_hidden_layers=n_hidden_layers_mlp, output_dim=1)
    elif model == "glovernnmlp":
        model = nlp.MyMLPResidualNetworkWithGloveEmbeddingsRNN(hidden_dim=hidden_dim, glove_vectors=glove_vectors, n_layers_rnn=n_layers_rnn, n_hidden_layers_mlp=n_hidden_layers_mlp, output_dim=1)
    elif model == "glovelstmmlp-mean":
        model = nlp.MyMLPResidualNetworkWithGloveEmbeddingsLSTMMeanPooling(hidden_dim=hidden_dim, glove_vectors=glove_vectors, n_layers_rnn=n_layers_rnn, n_hidden_layers_mlp=n_hidden_layers_mlp, output_dim=1)
    elif model == "glovelstmmlp-c":
        model = nlp.MyMLPResidualNetworkWithGloveEmbeddingsLSTMLastState(hidden_dim=hidden_dim, glove_vectors=glove_vectors, n_layers_rnn=n_layers_rnn, n_hidden_layers_mlp=n_hidden_layers_mlp, output_dim=1)
    elif model == "glovelstmmlp-h":
        model = nlp.MyMLPResidualNetworkWithGloveEmbeddingsLSTMLastHidden(hidden_dim=hidden_dim, glove_vectors=glove_vectors, n_layers_rnn=n_layers_rnn, n_hidden_layers_mlp=n_hidden_layers_mlp, output_dim=1)


    model, losses = nlp.train_binary_model(model, dataset_train, n_epochs=n_epochs)
    console.print("Training finished")
    console.print(f"Loss: {losses[-1]}")
    accuracy = nlp.test_binary_model(model, dataset_test)
    console.print(f"Accuracy: {accuracy}")
    accuracy_train = nlp.test_binary_model(model, dataset_train)
    console.print(f"Accuracy in train set: {accuracy_train}")


@app.command('info')
def print_info(custom_message : str = ""):
    """
    Print information about the module
    """
    console.print("Hello! I am My NLP Library")
    console.print(f"Author: { nlp.__author__}")
    console.print(f"Version: { nlp.__version__}")
    if custom_message != "":
        console.print(f"Custom message: {custom_message}")


if __name__ == "__main__":
    app()