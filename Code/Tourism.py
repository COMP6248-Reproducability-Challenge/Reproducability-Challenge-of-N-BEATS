import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.models import NBEATSModel
from darts.metrics import mape, r2_score
import time

t0 = time.time()

def display_forecast(pred_series, ts_transformed, forecast_type, start_date=None):
    plt.figure(figsize=(8, 5))
    if start_date:
        ts_transformed = ts_transformed.drop_before(start_date)
    ts_transformed.univariate_component(0).plot(label="actual")
    pred_series.plot(label=("historic " + forecast_type + " forecasts"))
    plt.title(
        "R2: {}".format(r2_score(ts_transformed.univariate_component(0), pred_series))
    )
    plt.legend()
    
data=TimeSeries.from_csv('Tourism_data_quart.csv',time_col='Quart')

N_Beatsm=NBEATSModel(input_chunk_length=12,output_chunk_length=6)
N_Beatsm.fit(data)

N_Beatsm1=NBEATSModel(input_chunk_length=9,output_chunk_length=3,num_stacks=10,num_blocks=3,num_layers=4,layer_widths=512,
    n_epochs=100,
    nr_epochs_val_period=1,
    batch_size=600,
    model_name="nbeats_run",torch_device_str='cuda:0')
N_Beatsm1.fit(data)

N_Beatsm2=NBEATSModel(input_chunk_length=9,output_chunk_length=3,num_stacks=10,num_blocks=2,num_layers=4,layer_widths=512,
    n_epochs=100,
    nr_epochs_val_period=1,
    batch_size=400,
    model_name="nbeats_run",torch_device_str='cuda:0')
N_Beatsm2.fit(data)

N_Beatsm3=NBEATSModel(input_chunk_length=12,output_chunk_length=3,model_name="nbeats_run",torch_device_str='cuda:0')
N_Beatsm3.fit(data)

N_Beatsm4=NBEATSModel(input_chunk_length=6,output_chunk_length=3,num_stacks=5,num_blocks=2,num_layers=4,layer_widths=512,
    n_epochs=100,
    nr_epochs_val_period=1,
    batch_size=400,
    model_name="nbeats_run",torch_device_str='cuda:0')
N_Beatsm4.fit(data)

real_data=TimeSeries.from_csv('Tourism_realdata_quart.csv',time_col='Quart')

N_Beatsp=N_Beatsm.predict(6)
N_Beatsp1=N_Beatsm1.predict(6)
N_Beatsp2=N_Beatsm2.predict(6)
N_Beatsp3=N_Beatsm3.predict(6)
N_Beatsp4=N_Beatsm4.predict(6)

#Result of common model
N_Beatsp_itg_common = (N_Beatsp+N_Beatsp1+N_Beatsp2+N_Beatsp3+N_Beatsp4)/5

Mape_itg_common = mape(real_data,N_Beatsp_itg_common)

#recording running time using common model
t1 = time.time() - t0

#Interpretable model

#N_Beat_i1:input_chunk_length=9, batch_size=600
N_Beat_i1=NBEATSModel(input_chunk_length=9,output_chunk_length=3,num_stacks=10,num_blocks=2,num_layers=4,layer_widths=512,
    n_epochs=100,
    nr_epochs_val_period=1,
    batch_size=600,
    model_name="nbeats_run",torch_device_str='cuda:0')
N_Beat_i1.fit(data)

N_Beat_i1p = N_Beat_i1.predict(6)
Mape_i1 = mape(real_data, N_Beat_i1p)

#N_Beat_i2:input_chunk_length=12, batch_size=600
N_Beat_i2=NBEATSModel(input_chunk_length=12,output_chunk_length=3,num_stacks=10,num_blocks=2,num_layers=4,layer_widths=512,
    n_epochs=100,
    nr_epochs_val_period=1,
    batch_size=600,
    model_name="nbeats_run",torch_device_str='cuda:0')
N_Beat_i2.fit(data)

N_Beat_i2p = N_Beat_i2.predict(6)
Mape_i2 = mape(real_data, N_Beat_i2p)

#N_Beat_i3:input_chunk_length=6, batch_size=600
N_Beat_i3=NBEATSModel(input_chunk_length=6,output_chunk_length=3,num_stacks=10,num_blocks=2,num_layers=4,layer_widths=512,
    n_epochs=100,
    nr_epochs_val_period=1,
    batch_size=600,
    model_name="nbeats_run",torch_device_str='cuda:0')
N_Beat_i3.fit(data)

N_Beat_i3p = N_Beat_i3.predict(6)
Mape_i3 = mape(real_data, N_Beat_i3p)

#N_Beat_i4:input_chunk_length=9, batch_size=600
N_Beat_i4=NBEATSModel(input_chunk_length=9,output_chunk_length=3,num_stacks=10,num_blocks=2,num_layers=4,layer_widths=512,
    n_epochs=100,
    nr_epochs_val_period=1,
    batch_size=600,
    model_name="nbeats_run",torch_device_str='cuda:0')
N_Beat_i4.fit(data)

N_Beat_i4p = N_Beat_i4.predict(6)
Mape_i4 = mape(real_data, N_Beat_i4p)

#N_Beat_i5:input_chunk_length=6, batch_size=400
N_Beat_i5=NBEATSModel(input_chunk_length=3,output_chunk_length=3,num_stacks=10,num_blocks=2,num_layers=4,layer_widths=512,
    n_epochs=100,
    nr_epochs_val_period=1,
    batch_size=400,
    model_name="nbeats_run",torch_device_str='cuda:0')
N_Beat_i5.fit(data)

N_Beat_i5p = N_Beat_i5.predict(6)
Mape_i5 = mape(real_data, N_Beat_i5p)

#calculate the prediction results and mape value by integrated interpretable model
N_Beatsp_itg_extra = (N_Beat_i1p+N_Beat_i2p+N_Beat_i3p+N_Beat_i4p+N_Beat_i5p)/5

Mape_itg_extra = mape(real_data,N_Beatsp_itg_extra)

#recording running time using interpretable model
t2 = time.time() - t0 - t1