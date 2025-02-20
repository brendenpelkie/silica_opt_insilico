import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from silica_opt_insilico import experiment, data_processing


class ContourAnimation:
    def __init__(self, ammonia_range, water_range, teos_range, 
                 Z_ammonia, Z_water, data, best_uuid, n_batches, 
                 true_min, trial_name, n_grid, m_samples, batch_size):
        """
        Initializes the animated contour plot.

        Parameters:
        - ammonia_range, water_range, teos_range: Value ranges for the contours.
        - Z_ammonia, Z_water: Precomputed contour grid data.
        - batches_points: List of batches (each containing points to plot).
        - best_points: List of best points per batch.
        - batch_names: List of batch names.
        - true_min: True minimum point coordinates.
        - trial_name: Title of the animation.
        - n_batches: Number of batches.
        - n_grid: Grid size for contours.
        """
        self.data = data
        self.ammonia_range = ammonia_range
        self.water_range = water_range
        self.teos_range = teos_range
        self.Z_ammonia = Z_ammonia
        self.Z_water = Z_water
        self.true_min = true_min
        self.trial_name = trial_name
        self.n_batches = n_batches
        self.n_grid = n_grid
        self.m_samples = m_samples
        self.best_uuid = best_uuid
        self.n_sobol = 2**m_samples
        self.batch_size = batch_size


        batch_nums = [str(i+1) for i in range(self.n_batches)]
        self.batch_names = ['Sobol random']
        self.batch_names.extend(batch_nums)


        self.batches_points = self.get_batch_splits()
        self.best_points = self.get_best_points()

        self.num_frames = n_batches + 1  # Animation frames

        # Create figure with two subplots
        self.fig, self.ax = plt.subplots(1, 2, figsize=(13, 6))

        # Plot Contour 1: TEOS vs. Water
        contour1 = self.ax[0].contourf(self.teos_range, self.water_range, Z_ammonia, levels=20, cmap='viridis')
        self.fig.colorbar(contour1, ax=self.ax[0], label='AP_distance')
        self.ax[0].set_xlabel('TEOS')
        self.ax[0].set_ylabel('Water')
        self.ax[0].set_title('TEOS-Water')

        # Plot Contour 2: TEOS vs. Ammonia
        contour2 = self.ax[1].contourf(self.teos_range, self.ammonia_range, Z_water, levels=20, cmap='viridis')
        self.fig.colorbar(contour2, ax=self.ax[1], label='AP_distance')
        self.ax[1].set_xlabel('TEOS')
        self.ax[1].set_ylabel('Ammonia')
        self.ax[1].set_title('TEOS-Ammonia')

        # Highlight the true minimum
        self.ax[0].scatter(self.true_min[0], self.true_min[2], marker='*', color='pink')
        self.ax[1].scatter(self.true_min[0], self.true_min[1], marker='*', color='pink')

        self.fig.suptitle(self.trial_name)

        # Text for batch label
        self.text = self.ax[0].text(0.06, 0.135, '', fontsize=12, color="white", backgroundcolor="black")

        # List to track scatter points for updating
        self.scatter_plots = []

    def get_batch_num(self, i):
        if i < self.n_sobol:
            return 0
        else:
            return int(np.ceil((i - self.n_sobol + 1)/self.batch_size))

    def get_batch_splits(self):
        ## Get batches

        i = 0
        n_sobol = 2**self.m_samples
        batches_points = []
        batch = []
        for uuid_val, data_val in self.data.items():
            point = [data_val['teos_vol_frac'], data_val['ammonia_vol_frac'], data_val['water_vol_frac']]
            batch.append(point)
                
            if i == n_sobol -1:
                batches_points.append(batch)
                batch = []

            if (i- (n_sobol - 1)) % self.batch_size == 0 and i > n_sobol - 1:
                batches_points.append(batch)
                batch = []

            i += 1
        print(batches_points)
        return batches_points

    def get_best_points(self):
        best_points = {}

        for i, (uuid_val, data_val) in enumerate(self.data.items()):
            if uuid_val in self.best_uuid:
                # get batch num:
                batch_num = self.get_batch_num(i)
                point = [data_val['teos_vol_frac'], data_val['ammonia_vol_frac'], data_val['water_vol_frac']]
                best_points[batch_num] = point    

        return best_points    



    def get_alpha(self, i, frame):
        """Determines alpha transparency for previous points."""
        if frame - i >= 5:
            return 0.1
        if frame - i  == 0:
            return 1
        else:
            return 1 - 0.2 * (frame - i)

    def update(self, frame):
        """Update function for animation."""

        # Remove old scatter points
        for sc in self.scatter_plots:
            try:
                sc.remove()
            except:
                continue
        
        # Add new points for each batch
        for i in range(frame + 1):
            current_batch = self.batches_points[i]
            self.text.set_text('i: '+str(i)+'Batch ' + self.batch_names[i]+ 'frame '+str(frame)+str(len(current_batch)))
            alpha = self.get_alpha(i, frame)
            #current_batch = self.batches_points[i]
            color = 'red' if frame - i  == 0 else 'k'

            for point in current_batch:
                sc1 = self.ax[0].scatter(point[0], point[2], color=color, alpha=alpha)
                sc2 = self.ax[1].scatter(point[0], point[1], color=color, alpha=alpha)
                self.scatter_plots.append(sc1)
                self.scatter_plots.append(sc2)

            # Highlight best points in cyan
            try:
                bp = self.best_points[i]
                self.ax[0].scatter(bp[0], bp[2], color='cyan', marker='*', s=100)
                self.ax[1].scatter(bp[0], bp[1], color='cyan', marker='*', s=100)
            except:
                continue

        return self.scatter_plots + [self.text]

    def run(self):
        """Run the animation and display it inline in Jupyter Notebook."""
        self.ani = animation.FuncAnimation(self.fig, self.update, frames=self.num_frames, interval=800, blit=False)
        return HTML(self.ani.to_jshtml())

    def save(self, filename="animation.mp4", format="mp4"):
        """Save the animation as a file."""
        self.ani = animation.FuncAnimation(self.fig, self.update, frames=self.num_frames, interval=800, blit=False)

        if format == "mp4":
            self.ani.save(filename, writer=animation.FFMpegWriter(fps=10))
        elif format == "gif":
            self.ani.save(filename, writer=animation.PillowWriter(fps=10))
        elif format == "html":
            html_content = self.ani.to_jshtml()
            with open(filename, "w") as f:
                f.write(html_content)
        else:
            raise ValueError("Unsupported format. Use 'mp4', 'gif', or 'html'.")

        print(f"Animation saved as {filename}")

def contour_eval(teos, ammonia, water, target_I, q_grid, return_scatter = False, amplitude_weight = 0.1):

    #teos = sample[0]
    #ammonia = sample[1]
    #water = sample[2]

    noise_level = 0
    sample_point = (teos, ammonia, water)

    scattering, real_sample_point, diameter, pdi = experiment.run_experiment(sample_point, noise_level, 10**q_grid, experiment.sld_silica, experiment.sld_etoh)

    # Process measurement
    ap_dist, ap_dist_report, I_scaled = data_processing.process_measurement(scattering, target_I, q_grid, amplitude_weight)

    if return_scatter:
        return ap_dist, scattering
    else:
        return ap_dist
