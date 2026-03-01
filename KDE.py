import numpy as np
import scipy.stats
import scipy.io
from enum import Enum
import copy
from typing import List
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.ticker import LogFormatterMathtext, LogLocator
from sklearn.metrics import confusion_matrix

np.random.seed(42)

# to implement:

# parallelism with other algorithms # e.g. derivatives + kde diff
# clustering for 2 segment detection
# inverse kde for user feedback  # i think there is no need for this
# early anomaly detection recognition
# re-evaluate points


class AnomalyTypes(Enum):
    KDE_NotEnoughData = 1
    KDE_Anomaly = 2
    KDE_Anomaly_Omitted = 3
    KDE_Anomaly_UserFeedback = 4
    
class AnomalyEntity():
    def __init__(self, anomaly_type, context):
        self.anomaly_type = anomaly_type
        self.context = context

class Point:
    def __init__(self, x, y, t, period_count, is_anomaly):
        self.x = x
        self.y = y
        self.t = t
        self.period_count = period_count
        self.is_true_anomaly = is_anomaly
        self.anomaly_list = []

    def __repr__(self):
        return f"(x={self.x}, y={self.y}, t={self.t}, period_count={self.period_count}, is_true_anomaly={self.is_true_anomaly})"

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
        self.points = points

    def __repr__(self):
        return f"Period with {len(self.points)} points, period_count={self.points[0].period_count}"

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


def load_data():
    data = scipy.io.loadmat('data/40014_dataset.mat')
    absolute_time = np.array([int(i) for i in data['absolute_time'][0]]) 
    value = np.array([int(i) for i in data['value'][0]]) 
    phase = np.array([int(i) for i in data['phase'][0]]) 
    phase_normalized = np.array([i for i in data['phase_normalized'][0]], dtype=np.float32)
    period_length = np.array([int(i) for i in data['period_length'][0]])
    period_count = np.array([int(i) for i in data['period_count'][0]])

    raise ValueError("Data loading is currently disabled. Please enable it to load the dataset.")

    mask = value > 10500
    absolute_time = absolute_time[mask]
    value = value[mask]
    phase = phase[mask]
    phase_normalized = phase_normalized[mask]
    period_length = period_length[mask]
    period_count = period_count[mask]
    
    # add anomalies
    # is_anomaly = []
    # for i in range(len(y_t)):
    #     if np.random.rand() < 0.002:  # 0.2% chance of anomaly
    #         y_t[i] += np.random.normal(0, 1000)  # add noise to the point
    #         is_anomaly.append(True)
    #     else:
    #         is_anomaly.append(False)

    is_anomaly = []
    bin_number = 100
    bins = [[] for _ in range(bin_number)]
    bins_limits = [[i/bin_number*5783, (i+1)/bin_number*5783] for i in range(bin_number)]

    for i in range(len(absolute_time)):
        placed = False
        for b in range(bin_number):
            if bins_limits[b][0] <= x_t[i] < bins_limits[b][1]:
                bins[b].append(i)
                placed = True
                break
        if not placed:
            bins[-1].append(i)

    means = [0 for _ in range(bin_number)]
    std = [0 for _ in range(bin_number)]
    quartiles = [[0,0] for _ in range(bin_number)] 
    quartiles_index = [[0,0] for _ in range(bin_number)]
    for b in range(bin_number):
        bin_values = [y_t[i] for i in bins[b]]
        means[b] = np.mean(bin_values)
        std[b] = np.std(bin_values)
        quartiles[b] = [np.percentile(bin_values, 1), np.percentile(bin_values, 99)]
        quartiles_index[b] = [np.argmin(np.abs(bin_values - np.percentile(bin_values, 1))), np.argmin(np.abs(bin_values - np.percentile(bin_values, 99)))]

    #plt.figure(figsize=(10,6))
    k = 8
    #print("Chance of a anomaly:", bin_number*2*100/len(y_t), "%")
    
    is_anomaly = [False for _ in range(len(y_t))]
    for b in range(bin_number):

        is_anomaly[bins[b][quartiles_index[b][0]]] = True
        is_anomaly[bins[b][quartiles_index[b][1]]] = True

        #plt.scatter(x_t[bins[b][quartiles_index[b][0]]], y_t[bins[b][quartiles_index[b][0]]], color='red')
        #plt.scatter(x_t[bins[b][quartiles_index[b][1]]], y_t[bins[b][quartiles_index[b][1]]], color='green')

        with open("synthetic_anomalies1.txt", "a") as f:
            f.write(f"{x_t[bins[b][quartiles_index[b][0]]]}; {y_t[bins[b][quartiles_index[b][0]]]}\n")

        with open("synthetic_anomalies2.txt", "a") as f:
            f.write(f"{x_t[bins[b][quartiles_index[b][1]]]}; {y_t[bins[b][quartiles_index[b][1]]]}\n")

        y_t[bins[b][quartiles_index[b][0]]] = y_t[bins[b][quartiles_index[b][0]]] + np.random.normal(-(y_t[bins[b][quartiles_index[b][1]]]-y_t[bins[b][quartiles_index[b][0]]])/k, std[b]/k)
        y_t[bins[b][quartiles_index[b][1]]] = y_t[bins[b][quartiles_index[b][1]]] + np.random.normal((y_t[bins[b][quartiles_index[b][1]]]-y_t[bins[b][quartiles_index[b][0]]])/k, std[b]/k)

        with open("synthetic_anomalies3.txt", "a") as f:
            f.write(f"{x_t[bins[b][quartiles_index[b][0]]]}; {y_t[bins[b][quartiles_index[b][0]]]}\n")

        with open("synthetic_anomalies4.txt", "a") as f:
            f.write(f"{x_t[bins[b][quartiles_index[b][1]]]}; {y_t[bins[b][quartiles_index[b][1]]]}\n")

        #plt.scatter(x_t[bins[b][quartiles_index[b][0]]], y_t[bins[b][quartiles_index[b][0]]], color='black')
        #plt.scatter(x_t[bins[b][quartiles_index[b][1]]], y_t[bins[b][quartiles_index[b][1]]], color='yellow')

    #plt.show()

    print("Started")

    return [Point(x_t[i], y_t[i], actual_t[i], period_count[i], is_anomaly[i]) for i in range(len(x_t))]

class GridKernelDensityEstimation():
    def __init__(self, y_bottom, y_upper, precision, bandwidth, outlier_threshold, outlier_omission, aggregation_window, memory_size, min_points_for_PDF, min_aggregation_window_points_PDF, grid_size):

        # Graphical parameters
        self.precision = precision
        self.y_bottom = y_bottom  # y-axis bottom value (visualization purpose)
        self.y_upper = y_upper   # y-axis upper value (visualization purpose)
        if self.y_upper <= self.y_bottom:
            raise ValueError("y_upper must be greater than y_bottom")

        # Model parameters
        self.bandwidth = bandwidth
        self.outlier_threshold = outlier_threshold
        self.outlier_omission = outlier_omission
        self.aggregation_window = aggregation_window
        self.memory_size = memory_size
        self.min_points_for_PDF = min_points_for_PDF  # at least n for PDF estimation (ignores aggregation_window)
        self.min_aggregation_window_points_PDF = min_aggregation_window_points_PDF # at least n for PDF estimation (one or the other)
        self.grid_size = grid_size

        self.grid = [[] for _ in range(grid_size)] # memory of the model

        self.kde_models = {} # pdf hash table per time point

    def compute_window(self, x):
        window = []
        for j in range(-self.aggregation_window, self.aggregation_window+1):
            if x+j < self.grid_size:
                window.extend(self.grid[x+j])
            else:
                window.extend(self.grid[x+j-self.grid_size])
            

        window.sort(key=lambda p: p.t)
        window = window[-self.memory_size:] # keep only the last memory_size values / if window smaller then memory_size, keep all values
        return window

    def calculate_weights(self, window, x):
        list_of_weights = []
        for point in window: 
            list_of_weights.append(self.aggregation_window - np.minimum(np.abs(point.x - x), self.grid_size - np.abs(point.x - x)) + 1) # Further from x, lower the weight
        list_of_weights = np.array(list_of_weights, dtype=float)
        list_of_weights /= np.sum(list_of_weights)
        return list_of_weights

    def can_create_pdf(self, x, window):
        return len(self.grid[x]) >= self.min_points_for_PDF or len(window) > self.min_aggregation_window_points_PDF

    def update_pdf(self, x):
        window = self.compute_window(x) # Aggregate points near in x

        if self.can_create_pdf(x, window):
            self.kde_models[x] = scipy.stats.gaussian_kde([point.y for point in window], bw_method = self.bandwidth(window), weights = self.calculate_weights(window, x))
        else:
            if x in self.kde_models:
                del self.kde_models[x]

    def fit(self, dataset: Dataset):
        for point in dataset.iter_points():
            self.grid[point.x].append(point)

        for x in range(self.grid_size):
            self.update_pdf(x)

    def score(self, point):
        if point.x not in self.kde_models:
            return -1
        kde = self.kde_models[point.x]
        return kde.evaluate(point.y)[0]

    def process_new_data(self, dataset: Dataset):
        for period in dataset.iter_periods():
            points_to_update = []

            for point in period.points:
                score = self.score(point)

                if score == -1:  # Not enough data for a valid conclusion
                    dataset.add_anomaly(point.t, AnomalyEntity(AnomalyTypes.KDE_NotEnoughData, None))
                    points_to_update.append(point)
                elif score < self.outlier_threshold: 
                    if score > self.outlier_omission: # Outlier
                        points_to_update.append(point)
                        dataset.add_anomaly(point.t, AnomalyEntity(AnomalyTypes.KDE_Anomaly, copy.deepcopy(self.kde_models[point.x])))
                    else: # Outlier to be omitted
                        dataset.add_anomaly(point.t, AnomalyEntity(AnomalyTypes.KDE_Anomaly_Omitted, copy.deepcopy(self.kde_models[point.x])))
                else:  # Not an outlier
                    points_to_update.append(point)

            for point in points_to_update:
                self.grid[point.x].append(point)
                self.update_pdf(point.x)

        # for plots only 
        for x in range(self.grid_size):
            self.update_pdf(x)

    def plot_heatmap(self, fig_number, save, frame):
        y_value = np.linspace(self.y_bottom, self.y_upper, self.precision)
        heatmap = np.zeros((self.precision, self.grid_size))
    
        for x in sorted(self.kde_models.keys()):
            kde = self.kde_models[x]
            heatmap[:, x] = kde.evaluate(y_value)

        # Remove color from parts outside the omission threshold
        masked_heatmap = np.ma.masked_less(heatmap, self.outlier_omission)

        plt.figure(figsize=(16, 8))
        norm = PowerNorm(gamma=0.18, vmin=self.outlier_threshold, vmax=np.max(heatmap))
        img = plt.imshow(masked_heatmap, aspect='auto', origin='lower', extent=[0-0.5, self.grid_size-0.5, self.y_bottom, self.y_upper], norm=norm, cmap='turbo')
        
        # add log scale to colorbar
        log_ticks = [10**exp for exp in range(int(np.floor(np.log10(self.outlier_threshold))), int(np.ceil(np.log10(np.max(heatmap)))) + 1) if 10 ** exp >= self.outlier_threshold and 10 ** exp <= np.max(heatmap)]
        cbar = plt.colorbar(img, label='Probability Density', fraction=0.03, pad=0.01)
        cbar.set_ticks(log_ticks)
        cbar.set_ticklabels([tick for tick in log_ticks])
        cbar.locator = LogLocator(base=10.0, subs=(1.0, ), numticks=10)
        cbar.formatter = LogFormatterMathtext(base=10.0, labelOnlyBase=False)
        cbar.update_ticks()

        plt.xlabel('Phase of the period')
        plt.ylabel('Value')
        plt.xlim([0, self.grid_size])
        plt.ylim([self.y_bottom, self.y_upper])
        plt.title(f'PDF HeatMap, {frame}')
        plt.tight_layout()
        if save:
            plt.savefig(f"{"heatmap"}/plot_{fig_number:04d}.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_heatmap_with_points(self, dataset: Dataset, fig_number, save, frame):
        y_value = np.linspace(self.y_bottom, self.y_upper, self.precision)
        heatmap = np.zeros((self.precision, self.grid_size))
    
        for x in sorted(self.kde_models.keys()):
            kde = self.kde_models[x]
            heatmap[:, x] = kde.evaluate(y_value)

        # Remove color from parts outside the omission threshold
        masked_heatmap = np.ma.masked_less(heatmap, self.outlier_omission)

        plt.figure(figsize=(16, 6))


        normal_points = dataset.get_normal_points()
        plt.scatter([point.x for point in normal_points], [point.y for point in normal_points], color='green', edgecolors='white', s=30, label="Normal")

        anomaly_points = dataset.get_anomalous_points(AnomalyTypes.KDE_Anomaly)
        anomaly_points += dataset.get_anomalous_points(AnomalyTypes.KDE_Anomaly_Omitted)
        anomaly_points += dataset.get_anomalous_points(AnomalyTypes.KDE_Anomaly_UserFeedback)
        plt.scatter([point.x for point in anomaly_points], [point.y for point in anomaly_points], color='red', edgecolors='white', s=30, label="Anomaly")

        not_enough_data_points = dataset.get_anomalous_points(AnomalyTypes.KDE_NotEnoughData)
        plt.scatter([point.x for point in not_enough_data_points], [point.y for point in not_enough_data_points], color='orange', edgecolors='white', s=30, label="Undefined KDE")

        true_anomalies = [point for point in dataset.iter_points() if point.is_true_anomaly]
        plt.scatter([point.x for point in true_anomalies], [point.y for point in true_anomalies], facecolors='none', edgecolors='black', s=30, label="True Anomalies")

        # Heatmap
        norm = PowerNorm(gamma=0.18, vmin=self.outlier_threshold, vmax=np.max(heatmap))
        img = plt.imshow(masked_heatmap, aspect='auto', origin='lower', extent=[0-0.5, self.grid_size-0.5, self.y_bottom, self.y_upper], norm=norm, cmap='turbo')
        
        # add log scale to colorbar
        log_ticks = [10**exp for exp in range(int(np.floor(np.log10(self.outlier_threshold))), int(np.ceil(np.log10(np.max(heatmap)))) + 1) if 10 ** exp >= self.outlier_threshold and 10 ** exp <= np.max(heatmap)]
        cbar = plt.colorbar(img, label='Probability Density', fraction=0.03, pad=0.01)
        cbar.set_ticks(log_ticks)
        cbar.set_ticklabels([tick for tick in log_ticks])
        cbar.locator = LogLocator(base=10.0, subs=(1.0, ), numticks=10)
        cbar.formatter = LogFormatterMathtext(base=10.0, labelOnlyBase=False)
        cbar.update_ticks()

        plt.xlabel('Phase in the orbital period [s]')
        plt.ylabel('Battery Voltage [mV]')
        plt.xlim([0, self.grid_size])
        plt.ylim([self.y_bottom, self.y_upper])
        plt.set_cmap('turbo')
        plt.legend(loc="upper center", bbox_to_anchor=(0.45, 1.0))
        plt.title(f'PFKDE Heatmap, {frame}')
        plt.tight_layout()
        if save:
            plt.savefig(f"thesis2/frames/plot_{fig_number:04d}.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_pdf(self, jump, fig_number, save):
        y_values = np.linspace(self.y_bottom, self.y_upper, self.precision)
        
        for i, x in enumerate(sorted(self.kde_models.keys())):
            if i % jump != 0:
                continue

            kde = self.kde_models[x]

            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(y_values, kde.evaluate(y_values), label='KDE', color='blue')
            ax1.set_yscale('log')
            ax1.set_ylim([1e-13, 1])
            ax1.set_xlabel('Values')
            ax1.set_ylabel('Log[Score]', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_xlim([self.y_bottom, self.y_upper])

            # Plot the thresholds
            ax1.axhline(self.outlier_threshold, color='green', linestyle='--', label='Anomaly Threshold')
            ax1.axhline(self.outlier_omission, color='red', linestyle='--', label='Omission Threshold')

            window = self.compute_window(x)
            window = [point.y for point in window]
            
            # Plot histogram on the secondary y-axis
            ax2 = ax1.twinx()
            ax2.hist(window, bins=15, alpha=0.5, label='Histogram', color='gray')
            ax2.set_ylabel('Frequency', color='gray')
            ax2.tick_params(axis='y', labelcolor='gray')
            ax2.set_xlim([self.y_bottom, self.y_upper])

            plt.title(f'PDF at time {x}')
            fig.legend(loc='upper right')
            fig.tight_layout()

            if save:
                plt.savefig(f"frames/plot_{fig_number:04d}.png")
                plt.close()
            else:
                plt.show()

    @staticmethod
    def parameter_fitting(model, dataset: Dataset, initial_guess, scale_factor=2.0, min_value=None, max_value=None, tolerance=1e-9, max_iter=50):

        def test_func(parameter):
            model_test = copy.deepcopy(model)
            dataset_test = copy.deepcopy(dataset)
            model_test.outlier_threshold = parameter
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

    def reevaluate_training_dataset(self, test_sets: List[Dataset], index_of_revaluation: int, fig_number: int, save: bool, frame: str):
        revaluation_points = []

        for index in range(index_of_revaluation+1):
            for point in test_sets[index].iter_points():
                if point.has_anomaly(AnomalyTypes.KDE_NotEnoughData):
                    self.grid[point.x].remove(point)
                    revaluation_points.append(point)

        for index in reversed(range(index_of_revaluation+1)):
            for point in reversed(test_sets[index].iter_points()):
                if point.has_anomaly(AnomalyTypes.KDE_NotEnoughData):
                    window = self.compute_window(point.x)
                    if self.can_create_pdf(point.x, window):
                        test_sets[index].remove_anomaly(point.t, AnomalyTypes.KDE_NotEnoughData)
                        temp_kde = scipy.stats.gaussian_kde([p.y for p in window], bw_method=self.bandwidth(window), weights=self.calculate_weights(window, point.x))
                        score = temp_kde.evaluate(point.y)[0]

                        if score < self.outlier_threshold: 
                            if score > self.outlier_omission: # Outlier
                                self.grid[point.x].append(point)
                                test_sets[index].add_anomaly(point.t, AnomalyEntity(AnomalyTypes.KDE_Anomaly, copy.deepcopy(temp_kde)))
                            else: # Outlier to be omitted
                                test_sets[index].add_anomaly(point.t, AnomalyEntity(AnomalyTypes.KDE_Anomaly_Omitted, copy.deepcopy(temp_kde)))
                        else:  # Not an outlier
                            self.grid[point.x].append(point)
                    else:
                        self.grid[point.x].append(point)

        for x in range(self.grid_size):
            self.update_pdf(x)

        if save == False:
            return

        self.plot_heatmap_with_points(Dataset(revaluation_points), fig_number, save, frame)

    def plot_confusion_matrix_and_heatmap(self, test_sets: List[Dataset], epochs: int, save, fig_number, val):
        list_is_true_anomaly = []
        list_was_flagged = []
        for i in range(epochs):
            for each_point in test_sets[i].iter_points():
                list_is_true_anomaly.append(each_point.is_true_anomaly)
                if each_point.has_anomaly(AnomalyTypes.KDE_Anomaly) or each_point.has_anomaly(AnomalyTypes.KDE_Anomaly_Omitted) or each_point.has_anomaly(AnomalyTypes.KDE_Anomaly_UserFeedback):
                    list_was_flagged.append(True)
                else:
                    list_was_flagged.append(False)

        # Calculate confusion matrix
        cm = confusion_matrix(list_is_true_anomaly, list_was_flagged, labels=[False, True])
        tn, fp, fn, tp = cm.ravel()
        if (tp + fn) > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0
        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0

        print(f"{val:.16f};\t{tp};\t{fp};\t{tn};\t{fn};\t{recall:.4f};\t{precision:.4f};")
        
        if save == False:
            return

        temp_x = []
        temp_y = []
        temp_is_true_anomaly = []
        temp_was_flagged = []
        for i in range(epochs):
            for each_point in test_sets[i].iter_points():
                temp_x.append(each_point.x)
                temp_y.append(each_point.y)
                temp_is_true_anomaly.append(each_point.is_true_anomaly)
                if each_point.has_anomaly(AnomalyTypes.KDE_Anomaly) or each_point.has_anomaly(AnomalyTypes.KDE_Anomaly_Omitted) or each_point.has_anomaly(AnomalyTypes.KDE_Anomaly_UserFeedback):
                    temp_was_flagged.append(True)
                else:
                    temp_was_flagged.append(False)

        y_min = min(temp_y) if min(temp_y) < self.y_bottom else self.y_bottom
        y_max = max(temp_y) if max(temp_y) > self.y_upper else self.y_upper

        y_value = np.linspace(self.y_bottom, self.y_upper, self.precision)
        heatmap = np.zeros((self.precision, self.grid_size))
    
        for x in sorted(self.kde_models.keys()):
            kde = self.kde_models[x]
            heatmap[:, x] = kde.evaluate(y_value)

        # Remove color from parts outside the omission threshold
        masked_heatmap = np.ma.masked_less(heatmap, self.outlier_omission)

        plt.figure(figsize=(16, 6))

        plt.scatter([x for i, x in enumerate(temp_x) if temp_is_true_anomaly[i]], [y for i, y in enumerate(temp_y) if temp_is_true_anomaly[i]], color='red', edgecolors='white', s=20, label='FN')
        plt.scatter([x for i, x in enumerate(temp_x) if temp_was_flagged[i] and not temp_is_true_anomaly[i]], [y for i, y in enumerate(temp_y) if temp_was_flagged[i] and not temp_is_true_anomaly[i]], color='orange', edgecolors='white', s=20, label='FP')
        plt.scatter([x for i, x in enumerate(temp_x) if temp_was_flagged[i] and temp_is_true_anomaly[i]], [y for i, y in enumerate(temp_y) if temp_was_flagged[i] and temp_is_true_anomaly[i]], color='green', edgecolors='white', s=20, label='TP')


        norm = PowerNorm(gamma=0.18, vmin=self.outlier_threshold, vmax=np.max(heatmap))
        img = plt.imshow(masked_heatmap, aspect='auto', origin='lower', extent=[0-0.5, self.grid_size-0.5, self.y_bottom, self.y_upper], norm=norm, cmap='turbo')
        
        # add log scale to colorbar
        log_ticks = [10**exp for exp in range(int(np.floor(np.log10(self.outlier_threshold))), int(np.ceil(np.log10(np.max(heatmap)))) + 1) if 10 ** exp >= self.outlier_threshold and 10 ** exp <= np.max(heatmap)]
        cbar = plt.colorbar(img, label='Probability Density', fraction=0.03, pad=0.01)
        cbar.set_ticks(log_ticks)
        cbar.set_ticklabels([tick for tick in log_ticks])
        cbar.locator = LogLocator(base=10.0, subs=(1.0, ), numticks=10)
        cbar.formatter = LogFormatterMathtext(base=10.0, labelOnlyBase=False)
        cbar.update_ticks()

        plt.xlabel('Phase in the orbital period [s]')
        plt.ylabel('Battery Voltage [mV]')
        plt.xlim([0, self.grid_size])
        plt.ylim([y_min, y_max])
        plt.legend(loc="upper center", bbox_to_anchor=(0.45, 1.0))
        plt.title(f'Phase-Folded KDE Heatmap with Anomaly Classification Results')
        plt.tight_layout()
        
        if save:
            plt.savefig(f"thesis2/frames/plot_{fig_number:04d}.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()




