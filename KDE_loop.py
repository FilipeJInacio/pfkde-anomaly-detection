import numpy as np
from KDE import PFKDE, Dataset
from KDE_run import load_data, add_synthetic_anomalies
import copy



if __name__ == "__main__":    
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

    print("Val,TP,FP,TN,FN,Recall,Precision,F1")
    values = np.array([10**-1,10**-2, 10**-3, 10**-4, 10**-5, 10**-6, 10**-7, 10**-8, 10**-9, 10**-10, 10**-11, 10**-12, 10**-13, 10**-14, 10**-15])
    elements = np.array([0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1])
    thresholds = np.concatenate([v * elements for v in values])

    list_of_points, max_length = load_data()
    list_of_points = add_synthetic_anomalies(list_of_points, max_length, 100, 8)
    epochs = 50
    save = True

    for threshold in thresholds:
        dataset = Dataset(copy.deepcopy(list_of_points))
        test_sets = dataset/epochs # Split the test set into 50 equal parts
        
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







