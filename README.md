# CalorieTracker Model (Django)

This repository wraps your model training code in a minimal Django project.

Quickstart (macOS, zsh):

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure the dataset CSV is at `src/main/resources/diet_recommendations_dataset.csv`.

4. Run the training management command:

```bash
python manage.py train_model
```

Outputs (after run): `modelapp/output/model.h5` and `modelapp/output/loss.png`.
# calorietracker-model