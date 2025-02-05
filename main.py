from scipy.optimize import approx_fprime  # (For scalar functions; see note below.)
import matplotlib.pyplot as plt
import numpy as np
import time

perf_times = []

class NMesh:
    def __init__(self, f, bounds, resolution, dimensions):
        if len(bounds) != dimensions:
            raise Exception("Bounds and dimensions must match")
        self.__bounds = bounds
        self.__dimension = dimensions
        self._mesh = np.meshgrid(
            *self.__generate_spaces(bounds, resolution, dimensions)
        )
        self.__samples = np.array([self.__sample_point() for _ in range(1000)])
        self._f = f

    def __generate_spaces(self, bounds, resolution, n_dimensions):
        return [
            np.linspace(bounds[indx][0], bounds[indx][1], resolution)
            for indx in range(n_dimensions)
        ]

    def eval_jacobian(self, points):
        # Using an adaptive finite difference step:
        scale = np.linalg.norm(points)
        if scale < 1e-12:
            scale = 1.0
        epsilon_step = 1e-6 * scale  # adjust as needed
        wrapped_f = lambda vec: self._f(*vec)
        # Note: approx_fprime is intended for scalar-valued functions.
        # For vector functions, consider using scipy.optimize.approx_derivative.
        return approx_fprime(points, wrapped_f, epsilon_step)

    def set_f(self, f):
        self._f = f

    def __sample_point(self):
        if self.__dimension == 1:
            return np.random.uniform(self.__bounds[0], self.__bounds[0], size=1)
        max_val, min_val = np.max(self.__bounds[:, 1]), np.min(self.__bounds[:, 0])
        return np.random.uniform(min_val, max_val, size=self.__dimension)

    def sample_point(self):
        return self.__samples[np.random.randint(0, len(self.__samples))]

    def sample_f(self):
        return self._f(*self.sample_point())

    """
    Iterates the manifold's sample points by the transform (f) provided 
    a number of (n)-times and returns the final points. The samples 
    are updated on the manifold as well. 
    """
    def iterate_by_f(self, n=10):
        mapped_points = []
        for _ in range(n):
            t_points = self._f(*self.__samples.T)
            mapped_points.append(t_points)
            self.__samples = t_points.T
        return np.array(mapped_points)

def HenonMap(x_n, y_n):
    a, b, c = 1.4, 0.3, 1.0
    x_val = 1 - a*(x_n**2) + b*y_n
    y_val = c*x_n
    # Clamp the values to avoid extreme overflow
    x_val = np.clip(x_val, -10, 10)
    y_val = np.clip(y_val, -10, 10)
    return np.array([x_val, y_val])

def experiment(n_samples=25, n_iterations=1000):
    mesh = NMesh(
        f=HenonMap,
        bounds=np.array([[-1.5, 1.5], [-1.5, 1.5]]),
        resolution=1000,
        dimensions=2,
    )

    # Lists for performance and measurement tracking:
    mu_arr = []
    expectation_flow = []
    mapped_points = []
    clockings = []
    
    # Initialize variables for Lyapunov exponent computation.
    lyapunov_sum = 0.0
    lyapunov_trace = []  # running average of the exponent

    for i in range(n_iterations):
        p_time_start = time.perf_counter()
        # ---------------------- Technique begins ----------------------
        # Compute the Jacobians at a number of sample points:
        sampled_jacobians = np.array(
            [np.abs(mesh.eval_jacobian(mesh.sample_point())) for _ in range(n_samples)]
        )
        mu_Ti = np.mean(sampled_jacobians)
        t_points = mesh.iterate_by_f(n=1)  # evolve the system one step
        
        # Compute a Jacobian for one representative sample:
        J = mesh.eval_jacobian(mesh.sample_point())
        # Compute the local expansion factor; for a 2D map this ideally is the spectral norm.
        # If J is a flattened array, this may be a rough approximation.
        local_expansion = np.linalg.norm(J, 2)
        # Compute the logarithmic expansion:
        lyap_increment = np.log(local_expansion)
        lyapunov_sum += lyap_increment
        # Running average (largest Lyapunov exponent estimate so far):
        lyapunov_trace.append(lyapunov_sum / (i + 1))
        
        expectation_flow.append(np.sum(mu_arr) / (i + 1))
        clockings.append(time.perf_counter() - p_time_start)
        # ---------------------- Technique ends ----------------------
        mu_arr.append(mu_Ti)
        mapped_points.append(t_points)
        print(f"Computation progress {np.round((i/n_iterations)*100, 2)}%", end="\r")
        
    return {
        "mapped_points": mapped_points,
        "n_iterations": n_iterations,
        "expectation_flow": expectation_flow,
        "n_samples": n_samples,
        "clockings": clockings,
        "lyapunov_trace": lyapunov_trace  # include the Lyapunov exponent trace
    }

def main():
    results = experiment()
    mapped_points = results["mapped_points"]
    n_iterations = results["n_iterations"]
    expectation_flow = results["expectation_flow"]
    n_samples = results["n_samples"]
    lyapunov_trace = results["lyapunov_trace"]

    # Example: compute sequential differences between mapped points (existing code)
    mapped_point_diffs = [
        np.linalg.norm((mapped_points[i + 1] - mapped_points[i]))
        for i in range(len(mapped_points) - 1)
    ]
    smoothing_window = 40
    expectation_flow_trend = np.convolve(expectation_flow, np.ones(smoothing_window)/smoothing_window, mode='valid')
    lyapunov_trace_trend = np.convolve(lyapunov_trace, np.ones(smoothing_window)/smoothing_window, mode='valid')

    # Plotting the various measures:
    plt.figure(figsize=(10, 6))
    # plt.title("Geometric Evolution Under Iterative Maps")
    # plt.xlabel("Transformation Iterations T^i")
    # plt.ylabel("Geometric Change by Averaged Jacobians / Iterations")
    # plt.plot(range(n_iterations), expectation_flow, label=f"Expectation Flow (T^{n_samples})", color="blue", linewidth=2)
    plt.plot(range(n_iterations), expectation_flow, label="Expectation Flow", color="blue", alpha=0.6)
    plt.plot(range(n_iterations), lyapunov_trace, label="Lyapunov Exponent", color="green", alpha=0.6)

    # Plot the trend lines; note that convolve with mode='valid' returns a shorter array,
    # so we align it with the x-axis by starting at smoothing_window-1.
    plt.plot(np.arange(smoothing_window-1, n_iterations), expectation_flow_trend, 
            label="Expectation Flow Trend", color="red", linestyle="--", linewidth=2)
    plt.plot(np.arange(smoothing_window-1, n_iterations), lyapunov_trace_trend, 
            label="Lyapunov Exponent Trend", color="orange", linestyle="--", linewidth=2)

    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title("Expectation Flow & Lyapunov Exponent Over Iterations")
    plt.legend()

    plt.figure(figsize=(10, 6))
    plt.title("Pointwise Differences Under Iterative Maps")
    plt.xlabel("Transformation Iterations T^i")
    plt.ylabel("Manifold Point Difference at Each Iteration")
    plt.plot(mapped_point_diffs, label=f"Mapped Point Differences Under (T^{n_samples})", alpha=0.5)
    plt.plot(np.arange(smoothing_window - 1, len(mapped_point_diffs)), np.convolve(mapped_point_diffs, np.ones(smoothing_window)/smoothing_window, mode="valid"), label="Moving Average of Point Diffs", color="red", linestyle="--", linewidth=2)
    plt.legend()

    plt.figure(figsize=(10, 6))
    plt.title("Lyapunov Exponent Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Estimated Largest Lyapunov Exponent")
    plt.plot(lyapunov_trace, label="Lyapunov Trace", color="green", linewidth=2)
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()
