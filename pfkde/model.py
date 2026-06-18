import scipy.stats
import copy
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.ticker import LogFormatterMathtext, LogLocator


def compute_slices(n, k):
    base = n // k
    remainder = n % k
    slices = []
    start = 0

    for i in range(k):
        size = base + (1 if i < remainder else 0)
        end = start + size
        slices.append((start, end))
        start = end

    return slices

def normalize(array):
    min_val = np.min(array)
    max_val = np.max(array)
    if max_val - min_val == 0:
        return np.zeros_like(array)  # Avoid division by zero, return an array of zeros
    return (array - min_val) / (max_val - min_val)

class Point:    
    def __init__(self, t, value, phase, period_index, score=None, label=None, is_true_anomaly=None):
        self.t = t
        self.value = value
        self.phase = phase
        self.period_index = period_index
        self.score = score
        self.label = label    # 1 for anomaly, 0 for normal
        self.is_true_anomaly = is_true_anomaly

def custom_bandwidth_function(window):
    window.sort(key=lambda p: p.value)
    max_gap = max(window[i+1].value - window[i].value for i in range(len(window)-1))
    if window[-1].value == window[0].value:
        raise ValueError ("All points in the window have the same value.")
    gap_factor = 1 + max_gap / (window[-1].value - window[0].value)
    bd = 0.2 * len(window) ** (-1 / 5) * gap_factor ** 3
    return bd

def custom_weight_function(window, center_phase, aggregation_window_size, number_of_bins):
    weights = []
    for point in window: 
        d = abs(point.phase - center_phase)     # Further from center, lower the weight
        d = min(d, number_of_bins - d)          # because of the phasefold, the end is the start and the start is the end
        weights.append(aggregation_window_size - d + 1) 
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()
    return weights

class PFKDE():
    def __init__(self, 
                 n_bins, 
                 omission_threshold, 
                 n_minimum_points, 
                 aggregation_window_size, 
                 memory_size, 
                 bandwidth_function, 
                 weight_function, 

                 threshold_type,
                 anomaly_threshold=None,
                 contamination=None,
                 labels=None,

                 plot=False, # Should we plot?
                 y_bottom=None, # For plotting purposes, the minimum value of the y axis (because points can be very far and distort the figure scope)
                 y_upper=None,  # For plotting purposes, the maximum value of the y axis (because points can be very far and distort the figure scope)
                 precision=None, # For plotting purposes, the number of points to evaluate the KDE in the y axis, more points means smoother heatmap but more computational cost
                 fig_path=None, # For plotting purposes, the path where to save the figures
                 frame_n=None, # For plotting purposes, the number of frames to divide the dataset in for plotting
                 ):
        
        self.n_bins = n_bins
        self.omission_threshold = omission_threshold
        self.n_minimum_points = n_minimum_points
        self.aggregation_window_size = aggregation_window_size
        self.memory_size = memory_size

        if bandwidth_function == 0:
            self.bandwidth_function = "scott"
        elif bandwidth_function == 1:
            self.bandwidth_function = custom_bandwidth_function
        else:
            raise ValueError(f"Unsupported bandwidth function: {bandwidth_function}")
        
        if weight_function == 0:
            self.weight_function = lambda window, center_phase, aggregation_window_size, number_of_bins: np.ones(len(window)) / len(window) # uniform weights
        elif weight_function == 1:
            self.weight_function = custom_weight_function
        else:
            raise ValueError(f"Unsupported weight function: {weight_function}")

        self.bins = [[] for _ in range(self.n_bins)] # memory of the model
        self.kde_models = {} # pdf hash table per time point
        self.decision_scores_ = []

        self.threshold_type = threshold_type
        if self.threshold_type == -1:
            # Means no thresholding
            if plot == True:
                raise ValueError("Plotting is not supported when threshold_type is -1.")
            self.find_threshold = lambda dataset: None
        elif self.threshold_type == 0:
            self.find_threshold = self.manual_threshold_definition
            if anomaly_threshold is None:
                raise ValueError("Anomaly threshold must be provided for manual threshold definition.")
        elif self.threshold_type == 1:
            self.find_threshold = self.find_threshold_from_contamination
            if contamination is None:
                raise ValueError("Contamination must be provided for contamination-based threshold definition.")
        elif self.threshold_type == 2:
            self.find_threshold = self.find_threshold_from_labels
            if labels is None:
                raise ValueError("Labels must be provided for labels-based threshold definition.")
        else:
            raise ValueError(f"Unsupported threshold definition method: {self.threshold_type}")
        self.anomaly_threshold = anomaly_threshold
        self.contamination = contamination

        self.labels = labels
    
        # For plotting purposes
        self.plot = plot
        if self.plot:
            if labels is None or y_bottom is None or y_upper is None or precision is None or fig_path is None or frame_n is None:
                raise ValueError("labels, y_bottom, y_upper, precision, fig_path and frame_n must be provided for plotting.")
                
            self.y_bottom = y_bottom
            self.y_upper = y_upper
            self.precision = precision
            self.fig_number = 0
            self.fig_path = fig_path
            self.frame_i = 0
            self.frame_n = frame_n
            self.data_slices = None

            self.x_label = "Phase in the orbital period [s]"
            self.y_label = "Battery Voltage [mV]"

            if not os.path.exists(self.fig_path):
                os.makedirs(self.fig_path)

            self.plotting_memory = [] 
            # Because the anomaly threshold is applied only at the end of the dataset, the plots can only be done correctly at the end of the dataset
            #! This implies that the visualization is not online, which is untrue
            #! If a anomaly threshold is fixed at the start by using "manual", the plots can be done online, but the anomaly threshold may not be optimal, which is a problem for the visualization in the paper


        self.data = None
        self.reevaluation_points = None

    def format_data(self, data):
        aux_period = []
        aux = [Point(*data[0])]
        for i in range(1, len(data)):
            if data[i][3] > data[i-1][3]:  # period_index
                aux_period.append(aux)
                aux = []
            aux.append(Point(*data[i]))
        aux_period.append(aux)

        # if plotting is allowed, prepare partitions of the dataset for plotting
        # divide the aux_period by frame_n and save the index intervals for slicing, len(aux_period)/frame_n
        if self.labels is not None:
            # add labels into the points
            counter = 0
            for period in aux_period:
                for point in period:
                    point.is_true_anomaly = self.labels[counter] == 1
                    counter += 1
        
        if self.plot:
            self.data_slices = compute_slices(len(aux_period), self.frame_n)

        return aux_period

    def compute_KDE(self, window, phase):
        return scipy.stats.gaussian_kde([point.value for point in window], bw_method = self.bandwidth_function(window), weights = self.weight_function(window, phase, self.aggregation_window_size, self.n_bins))

    def compute_window(self, phase):
        window = []
        for j in range(-self.aggregation_window_size, self.aggregation_window_size+1):
            if phase+j < self.n_bins:
                window.extend(self.bins[phase+j])
            else:
                window.extend(self.bins[phase+j-self.n_bins])
            
        window.sort(key=lambda p: p.t)
        window = window[-self.memory_size:] # keep only the last memory_size values / if window smaller then memory_size, keep all values
        return window

    def can_create_pdf(self, phase, window):
        return len(window) >= self.n_minimum_points

    def update_pdf(self, phase):
        window = self.compute_window(phase) # Aggregate points near in phase

        if self.can_create_pdf(phase, window):
            self.kde_models[phase] = self.compute_KDE(window, phase)
        else:
            if phase in self.kde_models:
                del self.kde_models[phase] # An existing KDE is no longer valid because there are not enough points, so we remove it.

    def score(self, point):
        if point.phase not in self.kde_models:
            return None
        kde = self.kde_models[point.phase]
        point.score = kde.evaluate(point.value)[0]

    def update_PFKDE(self, points_to_update):
        for point in points_to_update:
            self.bins[point.phase].append(point) # Add point to memory

            # The addition of the point doesn't only affect the KDE of its own phase, but also the KDEs of the phases in the aggregation window range
            # Comment/Uncomment this part depending or performance needs.

            # self.update_pdf(point.phase)
            # or
            #! This may add slight improvement to the cost of a huge overhead
            for j in range(-self.aggregation_window_size, self.aggregation_window_size+1):
                if point.phase+j < self.n_bins:
                    self.update_pdf(point.phase+j)
                else:
                    self.update_pdf(point.phase+j-self.n_bins)

    def reevaluate_points(self):
        if len(self.reevaluation_points) > 0: # If there is at least one point
            if self.plot:
                self.save_context([self.reevaluation_points], self.kde_models, "before_reevaluation")

            # Remove all non-evaluated points from the KDE's memories
            for point in self.reevaluation_points:
                self.bins[point.phase].remove(point)

            # Update all KDEs
            for phase in range(self.n_bins):
                self.update_pdf(phase)

            # In reverse order
            for point in reversed(self.reevaluation_points):
                self.score(point)

                if point.score is None:
                    pass

                elif point.score < self.omission_threshold: # Is it to be omitted?
                    continue # Outlier to be omitted, don't add it to the KDE

                self.update_PFKDE([point]) # They are added pointwise because there is no notion of period here

            if self.plot:
                self.save_context([self.reevaluation_points], self.kde_models, "after_reevaluation")

            self.reevaluation_points = [point for point in self.reevaluation_points if point.score is None]

    def fit(self, data):
        # data: (n, 4), UNIX_timestamp, value, phase, period_index
        self.data = self.format_data(data)

        self.reevaluation_points = []

        for i, period in enumerate(self.data):
            if i % 1000 == 0: # Each 1000 periods, reevaluate points
                self.reevaluate_points() 

            points_to_update = []
            for point in period:
                # Normally, here would be the search for the bin that the point belongs to. 
                # Because of performance reasons and because the period only changes in length 5 seconds throughout the dataset
                # Instead of using normalized phase, we use the phase in seconds, which we use as an index to fasten the computations.
                self.score(point)

                if point.score is None:  # Not enough data for a valid conclusion
                    self.reevaluation_points.append(point) # Could be done by searching for None's in the data, but this way it is faster

                elif point.score < self.omission_threshold: # Is it to be omitted?
                    continue # Outlier to be omitted, don't add it to the KDE
                
                points_to_update.append(point)
            
            self.update_PFKDE(points_to_update) # KDE only updates at the end of the period

            if self.plot:
                if i == self.data_slices[self.frame_i][1] - 1: # Index of the last period to make the plot
                    self.save_context(self.data[self.data_slices[self.frame_i][0]:self.data_slices[self.frame_i][1]], self.kde_models, "after_PFKDE_update")
                    self.frame_i += 1

        self.reevaluate_points()

        # All points not evaluated can be considered anomalies of 0 density
        for period in self.data:
            for point in period:
                if point.score is None:
                    point.score = 0

        self.decision_scores_ = -np.log(np.array([point.score for period in self.data for point in period]) + 1e-20) # For time eval # Log because 1 is anomaly in timeeval but 0 is anomaly in our case
        
        self.threshold = self.find_threshold(self.data)
        print(f"Chosen threshold: {self.threshold}")

        if self.threshold is None:
            return
         
        self.classify_anomalies(self.threshold, self.data)

        if self.plot:
            self.frame_i = 0 # reset frame index for plotting (was used before to count where we were in the dataset when saving contexts) is going to be used for the title frame_i/frame_n
            self.print_metrics()
            for i in range(len(self.plotting_memory)):
                self.plot_from_context(i)
            self.plot_final_heatmap_with_points()

    def manual_threshold_definition(self, dataset):
        return self.anomaly_threshold

    def find_threshold_from_contamination(self, dataset):
        dataset_len = sum(len(period) for period in dataset)
        not_identified_len = len(self.reevaluation_points)
        n_anomalies = int(self.contamination * dataset_len) + 1
        if not_identified_len >= n_anomalies:
            print(f"Warning: The number of points that couldn't be evaluated ({not_identified_len}) is greater than the expected number of anomalies ({n_anomalies}). The threshold will be set to the omission threshold.")
            return self.omission_threshold
        scores = np.array([point.score for period in dataset for point in period])
        return np.partition(scores, n_anomalies)[n_anomalies]

    def find_threshold_from_labels(self, dataset):
        scores = np.array([point.score for period in dataset for point in period])
        labels = np.array([point.is_true_anomaly for period in dataset for point in period])
        best_f1 = -1
        best_threshold = None

        for threshold in np.unique(scores):
            pred = scores < threshold
            TP = np.sum((pred == 1) & (labels == 1))
            FP = np.sum((pred == 1) & (labels == 0))
            FN = np.sum((pred == 0) & (labels == 1))
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold

    def classify_anomalies(self, threshold, dataset):
        for period in dataset:
            for point in period:
                if point.score is None:
                    pass
                elif point.score > threshold:
                    point.label = 0 # Normal
                else:
                    point.label = 1 # Anomaly

    def save_context(self, dataset, kde_models, plot_type):
        self.plotting_memory.append([copy.deepcopy(dataset), copy.deepcopy(kde_models), plot_type])

    def plot_from_context(self, plot_index):
        dataset, kde_models, plot_type = self.plotting_memory[plot_index]
        if plot_type == "after_PFKDE_update":
            self.plot_heatmap_with_points(dataset, kde_models, f'PFKDE Heatmap, {self.frame_i+1}/{self.frame_n} frames')
            self.frame_i += 1
        elif plot_type == "before_reevaluation":
            self.plot_heatmap_with_points(dataset, kde_models, f'PFKDE Heatmap before re-evaluation at frame {self.frame_i}')
        elif plot_type == "after_reevaluation":
            self.plot_heatmap_with_points(dataset, kde_models, f'PFKDE Heatmap after re-evaluation at frame {self.frame_i}')
        
        self.fig_number += 1

    def plot_heatmap_with_points(self, dataset, kde_models, title):
        possible_values = np.linspace(self.y_bottom, self.y_upper, self.precision)
        heatmap = np.zeros((self.precision, self.n_bins))
    
        for phase in sorted(kde_models.keys()):
            kde = kde_models[phase]
            heatmap[:, phase] = kde.evaluate(possible_values)

        # Remove color from parts outside the omission threshold
        masked_heatmap = np.ma.masked_less(heatmap, self.omission_threshold)

        # Label the dataset
        self.classify_anomalies(self.threshold, dataset)

        plt.figure(figsize=(16, 6))

        normal_points = [point for period in dataset for point in period if point.label == 0]
        plt.scatter([point.phase for point in normal_points], [point.value for point in normal_points], color='green', edgecolors='white', s=30, label="Normal")

        anomaly_points = [point for period in dataset for point in period if point.label == 1]
        plt.scatter([point.phase for point in anomaly_points], [point.value for point in anomaly_points], color='red', edgecolors='white', s=30, label="Anomaly")

        not_enough_data_points = [point for period in dataset for point in period if point.score is None]
        plt.scatter([point.phase for point in not_enough_data_points], [point.value for point in not_enough_data_points], color='orange', edgecolors='white', s=30, label="Undefined KDE")

        true_anomalies = [point for period in dataset for point in period if point.is_true_anomaly]
        plt.scatter([point.phase for point in true_anomalies], [point.value for point in true_anomalies], facecolors='none', edgecolors='black', s=30, label="True Anomalies")

        # Heatmap
        norm = PowerNorm(gamma=0.18, vmin=self.threshold, vmax=np.max(heatmap))
        img = plt.imshow(masked_heatmap, aspect='auto', origin='lower', extent=[0-0.5, self.n_bins-0.5, self.y_bottom, self.y_upper], norm=norm, cmap='turbo')
        
        # add log scale to colorbar
        log_ticks = [10**exp for exp in range(int(np.floor(np.log10(self.threshold))), int(np.ceil(np.log10(np.max(heatmap)))) + 1) if 10 ** exp >= self.threshold and 10 ** exp <= np.max(heatmap)]
        cbar = plt.colorbar(img, label='Probability Density', fraction=0.03, pad=0.01)
        cbar.set_ticks(log_ticks)
        cbar.set_ticklabels([tick for tick in log_ticks])
        cbar.locator = LogLocator(base=10.0, subs=(1.0, ), numticks=10)
        cbar.formatter = LogFormatterMathtext(base=10.0, labelOnlyBase=False)
        cbar.update_ticks()

        plt.xlabel('Phase in the orbital period [s]')
        plt.ylabel('Battery Voltage [mV]')
        plt.xlim([0, self.n_bins-1])
        plt.ylim([self.y_bottom, self.y_upper])
        plt.set_cmap('turbo')
        plt.legend(loc="upper center", bbox_to_anchor=(0.45, 1.0))
        plt.title(title)
        plt.tight_layout()
        plt.savefig(self.fig_path + f"/plot_{self.fig_number:04d}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_final_heatmap_with_points(self):
        possible_values = np.linspace(self.y_bottom, self.y_upper, self.precision)
        heatmap = np.zeros((self.precision, self.n_bins))
    
        for phase in sorted(self.kde_models.keys()):
            kde = self.kde_models[phase]
            heatmap[:, phase] = kde.evaluate(possible_values)

        # Remove color from parts outside the omission threshold
        masked_heatmap = np.ma.masked_less(heatmap, self.omission_threshold)

        plt.figure(figsize=(16, 6))

        TP_points = [point for period in self.data for point in period if point.label == 1 and point.is_true_anomaly == 1]
        plt.scatter([point.phase for point in TP_points], [point.value for point in TP_points], color='green', edgecolors='white', s=30, label="TP")

        FN_points = [point for period in self.data for point in period if point.label == 0 and point.is_true_anomaly == 1]
        plt.scatter([point.phase for point in FN_points], [point.value for point in FN_points], color='red', edgecolors='white', s=30, label="FN")

        FP_points = [point for period in self.data for point in period if point.label == 1 and point.is_true_anomaly == 0]
        plt.scatter([point.phase for point in FP_points], [point.value for point in FP_points], color='orange', edgecolors='white', s=30, label="FP")

        # Heatmap
        norm = PowerNorm(gamma=0.18, vmin=self.threshold, vmax=np.max(heatmap))
        img = plt.imshow(masked_heatmap, aspect='auto', origin='lower', extent=[0-0.5, self.n_bins-0.5, self.y_bottom, self.y_upper], norm=norm, cmap='turbo')
        
        # add log scale to colorbar
        log_ticks = [10**exp for exp in range(int(np.floor(np.log10(self.threshold))), int(np.ceil(np.log10(np.max(heatmap)))) + 1) if 10 ** exp >= self.threshold and 10 ** exp <= np.max(heatmap)]
        cbar = plt.colorbar(img, label='Probability Density', fraction=0.03, pad=0.01)
        cbar.set_ticks(log_ticks)
        cbar.set_ticklabels([tick for tick in log_ticks])
        cbar.locator = LogLocator(base=10.0, subs=(1.0, ), numticks=10)
        cbar.formatter = LogFormatterMathtext(base=10.0, labelOnlyBase=False)
        cbar.update_ticks()

        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.xlim([0, self.n_bins-1])
        plt.ylim([self.y_bottom, self.y_upper])
        plt.set_cmap('turbo')
        plt.legend(loc="upper center", bbox_to_anchor=(0.45, 1.0))
        plt.title(f'Phase-Folded KDE Heatmap with Anomaly Classification Results')
        plt.tight_layout()
        plt.savefig(self.fig_path + f"/plot_{self.fig_number:04d}.png", dpi=300, bbox_inches='tight')
        plt.close()

        self.fig_number += 1

    def print_metrics(self):
        TP = sum(1 for period in self.data for point in period if point.label == 1 and point.is_true_anomaly == 1)
        FP = sum(1 for period in self.data for point in period if point.label == 1 and point.is_true_anomaly == 0)
        TN = sum(1 for period in self.data for point in period if point.label == 0 and point.is_true_anomaly == 0)
        FN = sum(1 for period in self.data for point in period if point.label == 0 and point.is_true_anomaly == 1)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall =    TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
