from KDE import PFKDE, Dataset, Point
import scipy.io
import numpy as np

def load_data():
    data = scipy.io.loadmat('data/40014_dataset.mat')
    absolute_time = data['absolute_time'][0].astype(np.int64)
    value = data['value'][0].astype(np.int64)
    phase = data['phase'][0].astype(np.int64)
    #! Not using phase_normalized because the change in period length is 5 seconds during the 1 year period, which is not significant enough to affect the results
    #! The implementation is not very optimized already, and having to search the correct bin for each point in the phase_normalized would make it even worse
    #phase_normalized = data['phase_normalized'][0].astype(np.float64)
    period_length = data['period_length'][0].astype(np.int64)
    period_count = data['period_count'][0].astype(np.int64)
    max_length = min(np.max(period_length), np.max(phase) + 1)
    return [Point(phase[i], value[i], absolute_time[i], period_count[i], False, value[i]) for i in range(len(phase))], max_length

def add_synthetic_anomalies(dataset, max_length, bin_number, k):

    bins = [[] for _ in range(bin_number)]

    # Preliminary dataset verification
    for i in range(len(dataset)):
        idx = min(int(dataset[i].phase / max_length * bin_number), bin_number - 1)
        bins[idx].append(i)

    # Verify if all the bins have at least one point, otherwise the algorithm will fail
    for bin in bins:
        if len(bin) < 1:
            raise ValueError("One of the bins has no points, consider reducing the number of bins or check the data distribution.")
        if len(bin) == 1: # In case there is only 1 point, we need to duplicate it to be able to create an anomaly
            i = bin[0]
            dataset.append(Point(dataset[i].phase, dataset[i].y, dataset[i].t, dataset[i].period_count, False, dataset[i].y))
            print(f"Bin with only 1 point found at index {i}, duplicating the point to create an anomaly.")

    # Sort data by dataset[i].t
    dataset.sort(key=lambda p: p.t)

    # Recalculate because we might have added points
    bins = [[] for _ in range(bin_number)]
    for i in range(len(dataset)):
        idx = min(int(dataset[i].phase / max_length * bin_number), bin_number - 1)
        bins[idx].append(i)

    for bin in bins:
        values = np.array([dataset[i].y for i in bin])
        p1, p99 = np.percentile(values, [1, 99]) 
        idx_low = np.argmin(np.abs(values - p1)) 
        idx_high = np.argmin(np.abs(values - p99))

        if idx_low == idx_high:
            idx_high = (idx_low + 1) % len(bin) # Just to make sure we have two different points, the specific choice does not matter much

        dataset[bin[idx_low]].is_true_anomaly = True
        dataset[bin[idx_high]].is_true_anomaly = True
        spread = abs(dataset[bin[idx_high]].y - dataset[bin[idx_low]].y)
        mean_shift = max([spread / k, 10])
        variance = max([np.std(values) / k, 10]) # 0.01 V minimum

        dataset[bin[idx_low]].y += int(np.random.normal(-mean_shift, variance))
        dataset[bin[idx_high]].y += int(np.random.normal(mean_shift, variance))

        # Guarantee that the lower anomaly is lower than the higher anomaly, in case the random shift made them swap
        if dataset[bin[idx_low]].y > dataset[bin[idx_high]].y:
            dataset[bin[idx_low]].y, dataset[bin[idx_high]].y = dataset[bin[idx_high]].y, dataset[bin[idx_low]].y

    return dataset

if __name__ == "__main__":
    dataset, max_length = load_data()
    dataset = add_synthetic_anomalies(dataset, max_length, 100, 8)
    dataset = Dataset(dataset)
    epochs = 50
    test_sets = dataset/epochs # Split the test set into 50 equal parts
    
    def bandwidth_function(window):
        window.sort(key=lambda p: p.y)
        max_gap = max(window[i+1].y - window[i].y for i in range(len(window)-1))
        if window[-1].y == window[0].y:
            raise ValueError ("All points in the window have the same y value.")
        gap_factor = 1 + max_gap / (window[-1].y - window[0].y)
        bd = 0.2 * len(window) ** (-1 / 5) * gap_factor ** 3
        return bd

    def weight_function(window, center_phase, aggregation_window_size, number_of_bins):
        weights = []
        for point in window: 
            d = abs(point.phase - center_phase)     # Further from center, lower the weight
            d = min(d, number_of_bins - d)          # because of the phasefold, the end is the start and the start is the end
            weights.append(aggregation_window_size - d + 1) 
        weights = np.asarray(weights, dtype=float)
        weights /= weights.sum()
        return weights

    threshold = 8*10**-5
    
    model1 = PFKDE( number_of_bins=max_length,        
                    anomaly_threshold=threshold, 
                    omission_threshold=threshold*10**-2,
                    minimum_points=15,
                    aggregation_window_size=15, 
                    memory_size=300, 
                    bandwidth_function=bandwidth_function,
                    weight_function=weight_function,
                    y_bottom=10500,                         # y bottom limit for the plot
                    y_upper=13000,                          # y upper limit for the plot
                    precision=200)                          # how many points per KDE

    save = False
    total_anomalies = [0,0,0]
    image_counter = 0

    for i in range(epochs):
        model1.process_new_data(test_sets[i])

        model1.plot_heatmap_with_points(test_sets[i], fig_number=image_counter, save=save, frame=f"{i+1}/{epochs}")
        image_counter += 1

        if i == 5 or i == 11 or i == 17 or i == 23 or i == 29 or i == 35 or i == 41 or i == 47:
            model1.reevaluate_training_dataset(test_sets, i, fig_number=image_counter, save=save, frame=f"Reevaluation at {i+1}/{epochs}")
            image_counter += 1

    model1.plot_confusion_matrix_and_heatmap(test_sets, epochs, save, image_counter, threshold)




