# Fourier Transformation

- Fourier transformation is applied to capture periodic components or seasonality within time-series data.
  - `fourier_transform` feature contains information about the amplitudes of different frequency components, aiding in the identification and modeling of cyclic patterns in the time series.

```Python
# Applying Fourier transformation for capturing seasonality

from scipy.fft import fft

def apply_fourier_transform(data):
    values = data['target'].values
    fourier_transform = fft(values)
    data['fourier_transform'] = np.abs(fourier_transform)
    return data

# Applying Fourier transformation to the dataset
fourier_data = apply_fourier_transform(original_data)
```
