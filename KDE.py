import numpy as np
import scipy.stats
from enum import Enum
import copy
from typing import List
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.ticker import LogFormatterMathtext, LogLocator

np.random.seed(42)

SAVE_PATH = "data/frames/"

class AnomalyTypes(Enum):
    KDE_NotEnoughData = 1
    KDE_Anomaly = 2
    KDE_Anomaly_Omitted = 3
    
class AnomalyEntity():
    def __init__(self, anomaly_type, context):
        self.anomaly_type = anomaly_type
        self.context = context

class Point:
    def __init__(self, phase, y, t, period_count, is_anomaly, y_original):
        self.phase = phase
        self.y = y
        self.t = t
        self.period_count = period_count
        self.is_true_anomaly = is_anomaly
        self.y_original = y_original
        self.anomaly_list = []

    def __repr__(self):
        return f"(phase={self.phase}, y={self.y}, t={self.t}, period_count={self.period_count}, is_true_anomaly={self.is_true_anomaly}, y_original={self.y_original})"

    def add_anomaly(self, anomaly):
        self.anomaly_list.append(anomaly)

    def remove_anomaly(self, type):
        for anomaly in self.anomaly_list:
            if anomaly.anomaly_type == type:
                self.anomaly_list.remove(anomaly)
                break

    def has_anomaly(self, type):
        for anomaly in self.anomaly_list:
            if anomaly.anomaly_type == type:
                return True
        return False

class Period:
    def __init__(self, points: List[Point]):
        self.points = points     # The idea is that there is linking. If the class point is changed, the period of the point has the point updated
        self.period_count = points[0].period_count if points else -1

    def __repr__(self):
        return f"Period with {len(self.points)} points, period_count={self.period_count}"

class Dataset:
    def __init__(self, time_series: List[Point]):
        self.time_series = time_series
    
        aux_period_series = []
        aux = [time_series[0]]
        for i in range(1, len(time_series)):
            if time_series[i].period_count > time_series[i-1].period_count:
                aux_period_series.append(aux)
                aux = []
            aux.append(time_series[i])
        aux_period_series.append(aux)
        self.period_series = [Period(period) for period in aux_period_series]
        
        self.len = len(time_series)
        self.period_count = len(self.period_series)

    def get_point(self, point_index):
        return self.time_series[point_index]

    def get_period(self, period_index):
        return self.period_series[period_index]

    def iter_points(self):
        return self.time_series
            
    def iter_periods(self):
        return self.period_series

    def __truediv__(self, n):
        split_size = int(self.len / n)
        split_sets = []

        for i in range(n):
            start = i * split_size
            end = (i + 1) * split_size if i != n - 1 else self.len
            split_sets.append(Dataset(self.time_series[start:end]))
        
        return split_sets

    def split_dataset(self, n):

        total_size = self.len

        if 0 < n < 1:
            n = int(total_size * n)
        elif n >= total_size:
            raise ValueError("The test set size must be less than the total dataset size.")
        
        if n >= total_size:
            raise ValueError("The sum of test and validation set sizes must be less than the total dataset size.")

        split1_set = Dataset(self.time_series[:total_size - n])
        split2_set = Dataset(self.time_series[total_size - n:])

        return split1_set, split2_set

    def add_anomaly(self, t, anomaly):
        if t < self.time_series[0].t or t > self.time_series[-1].t:
            raise ValueError("Time point out of range")
        for point in self.time_series:
            if point.t == t:
                point.add_anomaly(anomaly)
                break

    def get_normal_points(self):
        normal_points = []
        for point in self.time_series:
            if not point.anomaly_list:
                normal_points.append(point)
        return normal_points

    def get_anomalous_points(self, type):
        anomalous_points = []
        for point in self.time_series:
            for anomaly in point.anomaly_list:
                if anomaly.anomaly_type == type:
                    anomalous_points.append(point)
                    break
        return anomalous_points

    def remove_anomaly(self, t, type):
        for point in self.time_series:
            if point.t == t:
                point.remove_anomaly(type)
                break

    def count_anomaly(self, type):
        count = 0
        for point in self.time_series:
            for anomaly in point.anomaly_list:
                if anomaly.anomaly_type == type:
                    count += 1
                    break
        return count

class PFKDE():
    def __init__(self, number_of_bins, anomaly_threshold, omission_threshold, minimum_points, aggregation_window_size, memory_size, bandwidth_function, weight_function, y_bottom, y_upper, precision):

        # Model parameters
        self.number_of_bins = number_of_bins
        self.anomaly_threshold = anomaly_threshold
        self.omission_threshold = omission_threshold
        self.minimum_points = minimum_points
        self.aggregation_window_size = aggregation_window_size
        self.memory_size = memory_size

        # Functions
        self.bandwidth_function = bandwidth_function
        self.weight_function = weight_function
        
        # Graphical parameters
        self.precision = precision
        self.y_bottom = y_bottom  # y-axis bottom value (visualization purpose)
        self.y_upper = y_upper   # y-axis upper value (visualization purpose)
        if self.y_upper <= self.y_bottom:
            raise ValueError("y_upper must be greater than y_bottom")

        self.bins = [[] for _ in range(self.number_of_bins)] # memory of the model
        self.kde_models = {} # pdf hash table per time point

    def compute_KDE(self, window, phase):
        return scipy.stats.gaussian_kde([point.y for point in window], bw_method = self.bandwidth_function(window), weights = self.weight_function(window, phase, self.aggregation_window_size, self.number_of_bins))

    def compute_window(self, phase):
        window = []
        for j in range(-self.aggregation_window_size, self.aggregation_window_size+1):
            if phase+j < self.number_of_bins:
                window.extend(self.bins[phase+j])
            else:
                window.extend(self.bins[phase+j-self.number_of_bins])
            
        window.sort(key=lambda p: p.t)
        window = window[-self.memory_size:] # keep only the last memory_size values / if window smaller then memory_size, keep all values
        return window

    def can_create_pdf(self, phase, window):
        return len(window) >= self.minimum_points

    def update_pdf(self, phase):
        window = self.compute_window(phase) # Aggregate points near in phase

        if self.can_create_pdf(phase, window):
            self.kde_models[phase] = self.compute_KDE(window, phase)
        else:
            if phase in self.kde_models:
                del self.kde_models[phase] # An existing KDE is no longer valid because there are not enough points, so we remove it.

    def score(self, point):
        if point.phase not in self.kde_models:
            return -1
        kde = self.kde_models[point.phase]
        return kde.evaluate(point.y)[0]

    def process_new_data(self, dataset: Dataset):
        for period in dataset.iter_periods(): # This is different from the paper. For performance reasons, we only update the KDEs after processing a whole period.
            points_to_update = []

            for point in period.points:
                
                # Normally, here would be the search for the bin that the point belongs to. 
                # Because of performance reasons and because the period only changes in length 5 seconds throughout the dataset
                # Instead of using normalized phase, we use the phase in seconds, which we use as an index to fasten the computations.

                score = self.score(point)

                if score == -1:  # Not enough data for a valid conclusion
                    dataset.add_anomaly(point.t, AnomalyEntity(AnomalyTypes.KDE_NotEnoughData, None))
                    points_to_update.append(point)
                elif score < self.anomaly_threshold: 
                    if score > self.omission_threshold: # Outlier
                        points_to_update.append(point)
                        dataset.add_anomaly(point.t, AnomalyEntity(AnomalyTypes.KDE_Anomaly, copy.deepcopy(self.kde_models[point.phase])))
                    else: # Outlier to be omitted
                        dataset.add_anomaly(point.t, AnomalyEntity(AnomalyTypes.KDE_Anomaly_Omitted, copy.deepcopy(self.kde_models[point.phase])))
                else:  # Not an outlier
                    points_to_update.append(point)

            for point in points_to_update:
                self.bins[point.phase].append(point) # Add point to memory

            for x in range(self.number_of_bins): # Update all KDEs. Because the new points can be in any of the KDEs' windows, we need to update them all.
                self.update_pdf(x)

    def plot_heatmap_with_points(self, dataset: Dataset, fig_number, save, frame):
        y_value = np.linspace(self.y_bottom, self.y_upper, self.precision)
        heatmap = np.zeros((self.precision, self.number_of_bins))
    
        for phase in sorted(self.kde_models.keys()):
            kde = self.kde_models[phase]
            heatmap[:, phase] = kde.evaluate(y_value)

        # Remove color from parts outside the omission threshold
        masked_heatmap = np.ma.masked_less(heatmap, self.omission_threshold)

        plt.figure(figsize=(16, 6))

        normal_points = dataset.get_normal_points()
        plt.scatter([point.phase for point in normal_points], [point.y for point in normal_points], color='green', edgecolors='white', s=30, label="Normal")

        anomaly_points = dataset.get_anomalous_points(AnomalyTypes.KDE_Anomaly)
        anomaly_points += dataset.get_anomalous_points(AnomalyTypes.KDE_Anomaly_Omitted)
        plt.scatter([point.phase for point in anomaly_points], [point.y for point in anomaly_points], color='red', edgecolors='white', s=30, label="Anomaly")

        not_enough_data_points = dataset.get_anomalous_points(AnomalyTypes.KDE_NotEnoughData)
        plt.scatter([point.phase for point in not_enough_data_points], [point.y for point in not_enough_data_points], color='orange', edgecolors='white', s=30, label="Undefined KDE")

        true_anomalies = [point for point in dataset.iter_points() if point.is_true_anomaly]
        plt.scatter([point.phase for point in true_anomalies], [point.y for point in true_anomalies], facecolors='none', edgecolors='black', s=30, label="True Anomalies")

        # Heatmap
        norm = PowerNorm(gamma=0.18, vmin=self.anomaly_threshold, vmax=np.max(heatmap))
        img = plt.imshow(masked_heatmap, aspect='auto', origin='lower', extent=[0-0.5, self.number_of_bins-0.5, self.y_bottom, self.y_upper], norm=norm, cmap='turbo')
        
        # add log scale to colorbar
        log_ticks = [10**exp for exp in range(int(np.floor(np.log10(self.anomaly_threshold))), int(np.ceil(np.log10(np.max(heatmap)))) + 1) if 10 ** exp >= self.anomaly_threshold and 10 ** exp <= np.max(heatmap)]
        cbar = plt.colorbar(img, label='Probability Density', fraction=0.03, pad=0.01)
        cbar.set_ticks(log_ticks)
        cbar.set_ticklabels([tick for tick in log_ticks])
        cbar.locator = LogLocator(base=10.0, subs=(1.0, ), numticks=10)
        cbar.formatter = LogFormatterMathtext(base=10.0, labelOnlyBase=False)
        cbar.update_ticks()

        plt.xlabel('Phase in the orbital period [s]')
        plt.ylabel('Battery Voltage [mV]')
        plt.xlim([0, self.number_of_bins-1])
        plt.ylim([self.y_bottom, self.y_upper])
        plt.set_cmap('turbo')
        plt.legend(loc="upper center", bbox_to_anchor=(0.45, 1.0))
        plt.title(f'PFKDE Heatmap, {frame}')
        plt.tight_layout()
        if save:
            plt.savefig(SAVE_PATH + f"plot_{fig_number:04d}.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_pdf(self, jump, fig_number, save):
        y_values = np.linspace(self.y_bottom, self.y_upper, self.precision)
        
        for i, phase in enumerate(sorted(self.kde_models.keys())):
            if i % jump != 0:
                continue

            kde = self.kde_models[phase]

            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(y_values, kde.evaluate(y_values), label='KDE', color='blue')
            ax1.set_yscale('log')
            ax1.set_ylim([1e-13, 1])
            ax1.set_xlabel('Values')
            ax1.set_ylabel('Log[Score]', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_xlim([self.y_bottom, self.y_upper])

            # Plot the thresholds
            ax1.axhline(self.anomaly_threshold, color='green', linestyle='--', label='Anomaly Threshold')
            ax1.axhline(self.omission_threshold, color='red', linestyle='--', label='Omission Threshold')

            window = self.compute_window(phase)
            window = [point.y for point in window]
            
            # Plot histogram on the secondary y-axis
            ax2 = ax1.twinx()
            ax2.hist(window, bins=15, alpha=0.5, label='Histogram', color='gray')
            ax2.set_ylabel('Frequency', color='gray')
            ax2.tick_params(axis='y', labelcolor='gray')
            ax2.set_xlim([self.y_bottom, self.y_upper])

            plt.title(f'PDF at phase {phase}')
            fig.legend(loc='upper right')
            fig.tight_layout()

            if save:
                plt.savefig(SAVE_PATH + f"plot_{fig_number:04d}.png", dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()

    def reevaluate_training_dataset(self, test_sets: List[Dataset], index_of_revaluation: int, fig_number: int, save: bool, frame: str):
        revaluation_points = []

        for index in range(index_of_revaluation+1):         # Per dataset used
            for point in test_sets[index].iter_points():    # Per point of the dataset
                if point.has_anomaly(AnomalyTypes.KDE_NotEnoughData):
                    self.bins[point.phase].remove(point)    # Remove all points from the KDE memory
                    revaluation_points.append(point)        # Used for plotting

        for index in reversed(range(index_of_revaluation+1)):           # Per dataset used, starting from the most recent one, to first one
            for point in reversed(test_sets[index].iter_points()):      # Per point analysed, starting from the most recent one, to the first one
                if point.has_anomaly(AnomalyTypes.KDE_NotEnoughData):   # Of the point labelled with NotEnoughData, we try to reevaluate them.
                    window = self.compute_window(point.phase)           # Need to make a new KDE (because we changed the memory in the previous loop)
                    if self.can_create_pdf(point.phase, window):
                        test_sets[index].remove_anomaly(point.t, AnomalyTypes.KDE_NotEnoughData)   # If its possible to create a KDE without the point, it is no more an initialization point
                        temp_kde = self.compute_KDE(window, point.phase)
                        score = temp_kde.evaluate(point.y)[0]            

                        if score < self.anomaly_threshold: 
                            if score > self.omission_threshold: # Outlier
                                self.bins[point.phase].append(point)
                                test_sets[index].add_anomaly(point.t, AnomalyEntity(AnomalyTypes.KDE_Anomaly, copy.deepcopy(temp_kde)))
                            else: # Outlier to be omitted
                                test_sets[index].add_anomaly(point.t, AnomalyEntity(AnomalyTypes.KDE_Anomaly_Omitted, copy.deepcopy(temp_kde)))
                        else:  # Not an outlier
                            self.bins[point.phase].append(point)
                    else:
                        self.bins[point.phase].append(point)            # Append the point again but still with the NotEnoughData flag.

        for x in range(self.number_of_bins):
            self.update_pdf(x)

        self.plot_heatmap_with_points(Dataset(revaluation_points), fig_number, save, frame) # Plot with revaluation results

    def plot_confusion_matrix_and_heatmap(self, test_sets: List[Dataset], save, fig_number, val):
        list_is_true_anomaly = []
        list_was_flagged = []
        for i in range(len(test_sets)):
            for point in test_sets[i].iter_points():
                list_is_true_anomaly.append(point.is_true_anomaly)
                if point.has_anomaly(AnomalyTypes.KDE_Anomaly) or point.has_anomaly(AnomalyTypes.KDE_Anomaly_Omitted):
                    list_was_flagged.append(True)
                else:
                    list_was_flagged.append(False)

        y_true = np.asarray(list_is_true_anomaly, dtype=bool)
        y_pred = np.asarray(list_was_flagged, dtype=bool)

        tp = np.sum(y_true & y_pred)
        tn = np.sum(~y_true & ~y_pred)
        fp = np.sum(~y_true & y_pred)
        fn = np.sum(y_true & ~y_pred)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)

        print(f"{val:.16f},{tp},{fp},{tn},{fn},{recall:.4f},{precision:.4f},{f1:.4f}")
        
        list_of_phases = []
        list_of_y = []
        for i in range(len(test_sets)):
            for each_point in test_sets[i].iter_points():
                list_of_phases.append(each_point.phase)
                list_of_y.append(each_point.y)

        y_min = min(list_of_y) if min(list_of_y) < self.y_bottom else self.y_bottom  #! TEST
        y_max = max(list_of_y) if max(list_of_y) > self.y_upper else self.y_upper   
 
        y_value = np.linspace(self.y_bottom, self.y_upper, self.precision)
        heatmap = np.zeros((self.precision, self.number_of_bins))
    
        for phase in sorted(self.kde_models.keys()):
            kde = self.kde_models[phase]
            heatmap[:, phase] = kde.evaluate(y_value)

        # Remove color from parts outside the omission threshold
        masked_heatmap = np.ma.masked_less(heatmap, self.omission_threshold)

        plt.figure(figsize=(16, 6))

        plt.scatter([phase for i, phase in enumerate(list_of_phases) if list_is_true_anomaly[i]], [y for i, y in enumerate(list_of_y) if list_is_true_anomaly[i]], color='red', edgecolors='white', s=20, label='FN')
        plt.scatter([phase for i, phase in enumerate(list_of_phases) if list_was_flagged[i] and not list_is_true_anomaly[i]], [y for i, y in enumerate(list_of_y) if list_was_flagged[i] and not list_is_true_anomaly[i]], color='orange', edgecolors='white', s=20, label='FP')
        plt.scatter([phase for i, phase in enumerate(list_of_phases) if list_was_flagged[i] and list_is_true_anomaly[i]], [y for i, y in enumerate(list_of_y) if list_was_flagged[i] and list_is_true_anomaly[i]], color='green', edgecolors='white', s=20, label='TP')


        norm = PowerNorm(gamma=0.18, vmin=self.anomaly_threshold, vmax=np.max(heatmap))
        img = plt.imshow(masked_heatmap, aspect='auto', origin='lower', extent=[0-0.5, self.number_of_bins-0.5, self.y_bottom, self.y_upper], norm=norm, cmap='turbo')
        
        # add log scale to colorbar
        log_ticks = [10**exp for exp in range(int(np.floor(np.log10(self.anomaly_threshold))), int(np.ceil(np.log10(np.max(heatmap)))) + 1) if 10 ** exp >= self.anomaly_threshold and 10 ** exp <= np.max(heatmap)]
        cbar = plt.colorbar(img, label='Probability Density', fraction=0.03, pad=0.01)
        cbar.set_ticks(log_ticks)
        cbar.set_ticklabels([tick for tick in log_ticks])
        cbar.locator = LogLocator(base=10.0, subs=(1.0, ), numticks=10)
        cbar.formatter = LogFormatterMathtext(base=10.0, labelOnlyBase=False)
        cbar.update_ticks()

        plt.xlabel('Phase in the orbital period [s]')
        plt.ylabel('Battery Voltage [mV]')
        plt.xlim([0, self.number_of_bins-1])
        plt.ylim([y_min, y_max])
        plt.legend(loc="upper center", bbox_to_anchor=(0.45, 1.0))
        plt.title(f'Phase-Folded KDE Heatmap with Anomaly Classification Results')
        plt.tight_layout()
        
        if save:
            plt.savefig(SAVE_PATH + f"plot_{fig_number:04d}.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def parameter_fitting(model: PFKDE, dataset: Dataset, initial_guess, scale_factor=2.0, min_value=None, max_value=None, tolerance=1e-9, max_iter=50):

    def test_func(parameter):
        model_test = copy.deepcopy(model)
        dataset_test = copy.deepcopy(dataset)
        model_test.anomaly_threshold = parameter
        model_test.process_new_data(dataset_test)
        return dataset_test.count_anomaly(AnomalyTypes.KDE_Anomaly) == 0 and dataset_test.count_anomaly(AnomalyTypes.KDE_Anomaly_Omitted) == 0

    # Verify initial guess correctness
    if test_func(initial_guess):
        low = initial_guess
        high = initial_guess * scale_factor
        # Expand upward until failure or until max_value reached
        while test_func(high):
            low = high
            high *= scale_factor
            if max_value is not None and high > max_value:
                high = max_value
                break
    else:
        # initial_guess fails: shrink downward
        high = initial_guess
        if min_value is None:
            low = initial_guess / scale_factor
            while not test_func(low) and low > 0:
                high = low
                low /= scale_factor
                if min_value is not None and low < min_value:
                    low = min_value
                    break
        else:
            low = min_value
        if not test_func(low):
            raise ValueError("Could not find a valid parameter range: all tested values fail.")

    # Bisection: test_func(low)=True, test_func(high)=False (or high==max_value)
    for i in range(max_iter):
        mid = (low + high) / 2.0
        if test_func(mid):
            low = mid
        else:
            high = mid
        
        if abs(high - low) <= tolerance:
            print(f"Converged after {i+1} iterations: low={low}, high={high}")
            break
    return low


