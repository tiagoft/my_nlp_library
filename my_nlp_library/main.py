import my_nlp_library as nlp
import os
from pathlib import Path
from rich.console import Console
import typer

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command('imdb')
def test_with_imdb():
    """
    Test baseline model with imdb
    """
    dataset_train, dataset_test, tokenizer = nlp.get_imdb_dataset()
    vocab_size = tokenizer.vocab_size
    model = nlp.MyClassifier(vocab_size=vocab_size, embedding_dim=50, output_dim=1)
    model, losses = nlp.train_binary_model(model, dataset_train)
    console.print("Training finished")
    console.print(f"Loss: {losses[-1]}")
    accuracy = nlp.test_binary_model(model, dataset_test)
    console.print(f"Accuracy: {accuracy}")


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