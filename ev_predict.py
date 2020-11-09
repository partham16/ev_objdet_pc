from src import Predicter

if __name__ == "__main__":
    predicter = Predicter()
    predicter.image_from_url(
        "https://www.autobuzzer.net/wp-content/uploads/2013/03/tips-to-bby-a-new-car.jpg"
    )
    img, pred = predicter.predict()
    print("prediction calculation done!")
