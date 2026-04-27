import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# Global forward time
tforward = np.arange(1, 1826)

k3 = 5e-6
k4 = 8e-6
w2 = 80
BHd = 0.03 / 365
B = 5e-8 / 365
Sd = 3750
IHd  = 0.2
S = 65000
I = 0.00047

def model_2(t, y, k):
    Ld  = 1020 / 365
    L= 1000 / 365
    md = 1 / (2 * 365)
    vHd = 0.1
    m = 1 / (65 * 365)
    v = 0.15

    
    Sd, IHd, S, I, C = y
    k3, k4, w2, BHd, B, Sd0, IHd0 = k

    dy = np.zeros(5)

    fIwt = k3 * np.sin(2 * np.pi * (t + w2) / 365) + k4
    

    dy[0] = Ld - (BHd * IHd * Sd) - (fIwt * Sd) - (md * Sd)
    dy[1] = (BHd * IHd * Sd) + (fIwt * Sd) - (md + vHd) * IHd
    dy[2] = L - ( B * IHd * S) - (m * S)
    dy[3] = ( B * IHd * S) - (m + v) * I
    dy[4] =( B * IHd * S)

    return dy


def model2(k, tdata):
    y0 = [k[5], k[6], 65000, 0.00047, 0.0]
    sol = solve_ivp(
        fun=lambda t, y: model_2(t, y, k),
        t_span=(tdata.min(), tdata.max()),
        y0=y0,
        t_eval=tdata,
        method="BDF",
        rtol=1e-8,
        atol=1e-10
    )

    if not sol.success:
        return np.full_like(tdata, np.inf)

    Y = sol.y.T
    return Y[:, 4]


def residuals(k, tdata, qdata):
    return model2(k, tdata) - qdata


def curve_fitting_model2_last():
    data = np.loadtxt("AFluDat05-09.txt")

    tdata = data[:, 0].astype(int)
    qdata = data[:, 1]

    k0 = k0 = np.array([
    5e-6,   # k3
    8e-6,   # k4
    80,     # w2
    0.03/365,  # BHd
    5e-8/365,  # B
    3750,   # Sd
    0.2     # IHd
    ])


    lower_bounds = [
        0,      # k3
        0,      # k4
        0,      # w2
        0,      # BHd
        0,      # B
        0,      # Sd
        0       # IHd
    ]
    upper_bounds = [
        1e-3,   # k3
        1e-3,   # k4
        365,    # w2
        1,    # BHd
        1e-5,   # B
        1e6,  # Sd
        1       # IHd
    ]

    result = least_squares(
        residuals,
        k0,
        bounds=(lower_bounds, upper_bounds),
        args=(tdata, qdata),
        x_scale = 'jac',
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
        verbose=2,
        max_nfev=10000
    )

    k = result.x

    sol = solve_ivp(
        fun=lambda t, y: model_2(t, y, k),
        t_span=(tforward[0], tforward[-1]),
        y0=[k[5], k[6], 65000, 0.00047, 0.0],
        t_eval=tforward,
        method="BDF",
        rtol=1e-8,
        atol=1e-10
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    Y = sol.y.T

    plt.figure(figsize=(8, 5))

    plt.plot(
        tdata,
        qdata,
        ".",
        color="red",
        markersize=10,
        label="Data"
    )

    plt.plot(
        tforward,
        Y[:, 4],
        "b-",
        linewidth=1.2,
        label="Model fit"
    )

    plt.xticks(
        [0, 365, 730, 1095, 1460, 1825],
        ["2005", "2006", "2007", "2008", "2009", "2010"]
    )

    plt.yticks([0.001, 0.002, 0.003, 0.004, 0.005])

    plt.xlabel("Time ")
    plt.ylabel("Cumulative number of human cases")
    plt.grid(axis="x")
    plt.box(True)
    plt.legend()
    plt.show()

    return k


if __name__ == "__main__":
    fitted_parameters = curve_fitting_model2_last()
    print("Fitted parameters:")
    print(fitted_parameters)