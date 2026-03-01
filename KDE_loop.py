import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from KDE import GridKernelDensityEstimation, Dataset, load_data
import copy


if __name__ == "__main__":
    def bandwidth(window):
        window.sort(key=lambda p: p.y)
        max_gap = max(window[i+1].y - window[i].y for i in range(len(window)-1))
        if window[-1].y == window[0].y:
            raise ValueError ("All points in the window have the same y value.")
        gap_factor = 1 + max_gap / (window[-1].y - window[0].y)
        bd = 0.2 * len(window) ** (-1 / 5) * gap_factor ** 3
        return bd

    print("Val;\tTP;\tFP;\tTN;\tFN;\tRecall;\tPrecision;")
    # original list
    values = np.array([10**-1,10**-2, 10**-3, 10**-4, 10**-5, 10**-6, 10**-7, 10**-8, 10**-9, 10**-10, 10**-11, 10**-12, 10**-13, 10**-14, 10**-15])
    # multiply each value by 10 elements
    elements = np.array([0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1])
    values = np.concatenate([v * elements for v in values])

    list_of_points = load_data()

    for val in values:
        dataset = Dataset(copy.deepcopy(list_of_points))
        epochs = 50
        test_sets = dataset/epochs # Split the test set into 90 equal parts

        model1 = GridKernelDensityEstimation(y_bottom=10500,
                                            y_upper=13000,
                                            precision=200,
                                            bandwidth=bandwidth, 
                                            outlier_threshold=val, 
                                            outlier_omission=val*10**-2, 
                                            aggregation_window=15, 
                                            memory_size=300, 
                                            min_points_for_PDF=5, 
                                            min_aggregation_window_points_PDF=15, 
                                            grid_size=5783)


        save = False
        #total_anomalies = [0,0,0]

        #model1.fit(train_set)  

        image_counter = 0
        #model1.plot_heatmap(image_counter, save=save, frame="Training Data")
        #image_counter += 1

        # model1.outlier_threshold = GridKernelDensityEstimation.parameter_fitting(model1, parameter_set, model1.outlier_threshold, scale_factor=2.0, min_value=10**-12, tolerance=1e-14, max_iter=100)
        # model1.outlier_omission = model1.outlier_threshold * 10**-2

        # model1.plot_pdf(jump=100, fig_number=image_counter, save=save)

        for i in range(epochs):
            model1.process_new_data(test_sets[i])


            #model1.plot_heatmap_with_points(test_sets[i], fig_number=image_counter, save=save, frame=f"{i+1}/{epochs}")
            #image_counter += 1

            if i == 5 or i == 11 or i == 17 or i == 23 or i == 29 or i == 35 or i == 41 or i == 47:
                #model1.plot_heatmap_with_points(test_sets[i], fig_number=image_counter, save=save, frame=f"{i+1}/{epochs}")
                model1.reevaluate_training_dataset(test_sets, i, fig_number=image_counter, save=save, frame=f"Reevaluation at {i+1}/{epochs}")
                #image_counter += 1

            # model1.pool_user_feedback(test_sets[i], frame=f"{i+1}/{epochs}")

            # model1.test_criteria(frame=f"{i+1}/{epochs}")
            # model1.synthetic_user_feedback(test_sets[i].get_point(-1).t,frame=f"{i+1}/{epochs}", num_of_points=100)

            #if i == 1:
                #model1.plot_heatmap(image_counter, save=save, frame=f"{i+1}/{epochs}")
                #model2.plot_heatmap(image_counter, save=save, frame=f"{i+1}/{epochs}")

            #total_anomalies[0] += test_sets[i].count_anomaly(AnomalyTypes.KDE_Anomaly)
            #total_anomalies[1] += test_sets[i].count_anomaly(AnomalyTypes.KDE_Anomaly_Omitted)
            #total_anomalies[2] += test_sets[i].count_anomaly(AnomalyTypes.KDE_NotEnoughData)
            #print(f"Processed {i+1}/{epochs}, {test_sets[i].len} points, {test_sets[i].period_count} periods, {test_sets[i].count_anomaly(AnomalyTypes.KDE_Anomaly)} anomalies, {test_sets[i].count_anomaly(AnomalyTypes.KDE_Anomaly_Omitted)} anomalies omitted, {test_sets[i].count_anomaly(AnomalyTypes.KDE_NotEnoughData)} not enough data points")

        model1.plot_confusion_matrix_and_heatmap(test_sets, epochs, save, image_counter, val)




