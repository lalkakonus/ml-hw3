from dataloader import TDataLoader
from fm import FM


def main():
    dataloader = TDataLoader()
    X_train, X_test, y_train, y_test = dataloader.load(load_preprocessed=False)
    model = FM(n_iter=16, latent_dimension=5, lambda_w=4, lambda_v=7)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    prediction[prediction > 5] = 5
    prediction[prediction < 0] = 0
    dataloader.save(prediction)


if __name__ == "__main__":
   main()
