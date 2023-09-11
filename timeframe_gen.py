from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Generate training windows. month_training_window is the length of each window in months
def generate_date_windows(start, end, month_training_window=6):
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")
    windows = []
    train_duration = 0
    test_duration = 0

    train_duration = relativedelta(months=+month_training_window)
    test_duration = relativedelta(days=+15)

    while start_date + train_duration + test_duration <= end_date:
        train_start = start_date
        train_end = start_date + train_duration
        test_start = train_end + timedelta(days=1)
        test_end = test_start + test_duration

        windows.append({
            "train_start_date": train_start.strftime("%Y-%m-%d"),
            "train_end_date": train_end.strftime("%Y-%m-%d"),
            "test_start_date": test_start.strftime("%Y-%m-%d"),
            "test_end_date": test_end.strftime("%Y-%m-%d")
        })
        start_date += relativedelta(days=+15)

    return windows


if __name__ == "__main__":
    start_date = "2017-01-01"
    end_date = "2020-12-31"
    timestep = 24

    windows = generate_date_windows(start_date, end_date, timestep)
    for window in windows:
        print(window)




