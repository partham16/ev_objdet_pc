from src import Trainer

if __name__ == "__main__":
    trainer = Trainer()
    trainer.fine_tune()
    trainer.save_model()
