from KDE import GridKernelDensityEstimation, Dataset, load_data

if __name__ == "__main__":
    dataset = Dataset(load_data())
    epochs = 50
    test_sets = dataset/epochs # Split the test set into 90 equal parts
    
    def bandwidth(window):
        window.sort(key=lambda p: p.y)
        max_gap = max(window[i+1].y - window[i].y for i in range(len(window)-1))
        if window[-1].y == window[0].y:
            raise ValueError ("All points in the window have the same y value.")
        gap_factor = 1 + max_gap / (window[-1].y - window[0].y)
        bd = 0.2 * len(window) ** (-1 / 5) * gap_factor ** 3
        return bd

    threshold = 8*10**-5
    
    model1 = GridKernelDensityEstimation(y_bottom=10500,
                                        y_upper=13000,
                                        precision=200,
                                        bandwidth=bandwidth, 
                                        outlier_threshold=threshold, 
                                        outlier_omission=threshold*10**-2, 
                                        aggregation_window=15, 
                                        memory_size=300, 
                                        min_points_for_PDF=5, 
                                        min_aggregation_window_points_PDF=15, 
                                        grid_size=5783)


    save = True
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




