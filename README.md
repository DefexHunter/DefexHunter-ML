# JM1 Software Defect Prediction API

Production-ready ML API for predicting software defects using NASA JM1 dataset.

## Features

- Data preprocessing pipeline
- Class balancing with NearMiss
- Multiple ML models:
  - Random Forest
  - Decision Tree
  - SVM
  - KNN
  - XGBoost
- FastAPI prediction service
- Docker support
- CI/CD ready

## How to run it locally:
- 1. First clone the repository: `git clone <repo-url>`
- 2. Second, navigate into the project folder: `cd DefexHunter-ML`
- 3. Install the dependencies: `pip install -r requirements.txt`
- 4. Place the dataset file (`BalancedPROMISE_union.csv`) inside `src/data/`
- 5. Add the `.env` file in the project root directory
- 6. Run the training script to train and save models: `python src/train.py src/data/jm1_csv.csv`
- 7. Make sure Docker Engine is running (via Docker Desktop)
- 8. Build the Docker image: `docker compose build api --no-cache`
- 9. Start the application: `docker compose up --build`
- 10. After that, you can test the API using this Postman collection: https://documenter.getpostman.com/view/53206285/2sBXwvHTxC