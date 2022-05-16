import skfuzzy as fuzz
import numpy as np
import matplotlib.pyplot as plt

min = 0
max_temp = 40
max_percentage = 100

zero_low = 0.25 * max_temp
low_medium = 0.5 * max_temp
medium_high = 0.75 * max_temp

zero_low_humidity = 0.25 * max_percentage
low_medium_humidity = 0.5 * max_percentage
medium_high_humidity = 0.75 * max_percentage

x_temp = np.arange(min, max_temp, 1)
x_humidity = np.arange(min, max_percentage, 1)
x_intensity = np.arange(min, max_percentage, 1)


temp_zero = fuzz.trimf(x_temp, [min, min, zero_low])
temp_low = fuzz.trimf(x_temp, [min, zero_low, low_medium])
temp_medium = fuzz.trimf(x_temp, [zero_low, low_medium, medium_high])
temp_high = fuzz.trimf(x_temp, [low_medium, medium_high, max_temp])
temp_max= fuzz.trimf(x_temp, [medium_high, max_temp, max_temp])

humidity_zero = fuzz.trimf(x_humidity, [min, min, zero_low_humidity])
humidity_low = fuzz.trimf(x_humidity, [min, zero_low_humidity, low_medium_humidity])
humidity_medium = fuzz.trimf(x_humidity, [zero_low_humidity, low_medium_humidity, medium_high_humidity])
humidity_high = fuzz.trimf(x_humidity, [low_medium_humidity, medium_high_humidity, max_percentage])
humidity_max= fuzz.trimf(x_humidity, [medium_high_humidity, max_percentage, max_percentage])

intensity_zero = fuzz.trimf(x_intensity, [min, min, zero_low_humidity])
intensity_low = fuzz.trimf(x_intensity, [min, zero_low_humidity, low_medium_humidity])
intensity_medium = fuzz.trimf(x_intensity, [zero_low_humidity, low_medium_humidity, medium_high_humidity])
intensity_high = fuzz.trimf(x_intensity, [low_medium_humidity, medium_high_humidity, max_percentage])
intensity_max= fuzz.trimf(x_intensity, [medium_high_humidity, max_percentage, max_percentage])

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

ax0.plot(x_temp, temp_zero, 'b', linewidth=1.5, label='zero')
ax0.plot(x_temp, temp_low, 'g', linewidth=1.5, label='low')
ax0.plot(x_temp, temp_medium, 'r', linewidth=1.5, label='medium')
ax0.plot(x_temp, temp_high, 'y', linewidth=1.5, label='high')
ax0.plot(x_temp, temp_max, 'brown', linewidth=1.5, label='max')
ax0.set_title('temperature')
ax0.legend()

ax1.plot(x_humidity, humidity_zero, 'b', linewidth=1.5, label='zero')
ax1.plot(x_humidity, humidity_low, 'g', linewidth=1.5, label='low')
ax1.plot(x_humidity, humidity_medium, 'r', linewidth=1.5, label='medium')
ax1.plot(x_humidity, humidity_high, 'y', linewidth=1.5, label='high')
ax1.plot(x_humidity, humidity_max, 'brown', linewidth=1.5, label='max')
ax1.set_title('humidity')
ax1.legend()

ax2.plot(x_intensity, intensity_zero, 'b', linewidth=1.5, label='zero')
ax2.plot(x_intensity, intensity_low, 'g', linewidth=1.5, label='low')
ax2.plot(x_intensity, intensity_medium, 'r', linewidth=1.5, label='medium')
ax2.plot(x_intensity, intensity_high, 'y', linewidth=1.5, label='high')
ax2.plot(x_intensity, intensity_max, 'brown', linewidth=1.5, label='max')
ax2.set_title('intensity')
ax2.legend()


plt.tight_layout()
plt.show()

temp = 37
humidity = 30

temp_level_zero = fuzz.interp_membership(x_temp, temp_zero, temp)
temp_level_low = fuzz.interp_membership(x_temp, temp_low, temp)
temp_level_medium = fuzz.interp_membership(x_temp, temp_medium, temp)
temp_level_high = fuzz.interp_membership(x_temp, temp_high, temp)
temp_level_max = fuzz.interp_membership(x_temp, temp_max, temp)

humidity_level_zero = fuzz.interp_membership(x_humidity, humidity_zero, humidity)
humidity_level_low = fuzz.interp_membership(x_humidity, humidity_low, humidity)
humidity_level_medium = fuzz.interp_membership(x_humidity, humidity_medium, humidity)
humidity_level_high = fuzz.interp_membership(x_humidity, humidity_high, humidity)
humidity_level_max = fuzz.interp_membership(x_humidity, humidity_max, humidity)

# print(f'{temp_low=}')
# print(f'{humidity_low}')
# print(f'{temp_level_low=}')
# print(f'{humidity_level_low=}')
active_rule1 = np.fmax(temp_level_low, humidity_level_low) # rule 1: low temp & low humidity
# print(active_rule1)
active_rule2 = np.fmax(temp_level_low, humidity_level_medium) # rule 2: low temp & medium humidity
active_rule3 = np.fmax(temp_level_low, humidity_level_high) # rule 3: low temp & high humidity

active_rule4 = np.fmax(temp_level_medium, humidity_level_low) # rule 4: medium temp & low humidity
active_rule5 = np.fmax(temp_level_medium, humidity_level_medium) # rule 5: medium temp & medium humidity
active_rule6 = np.fmax(temp_level_medium, humidity_level_high) # rule 6: medium temp & high humidity

active_rule7 = np.fmax(temp_level_high, humidity_level_low) # rule 7:  high temp & low humidity
active_rule8 = np.fmax(temp_level_high, humidity_level_medium) # rule 8:  high temp & medium humidity
active_rule9 = np.fmax(temp_level_high, humidity_level_high) # rule 9:  high temp & high humidity

active_rule10 = np.fmax(temp_level_max, humidity_level_low) # rule 10: max temp & low humidity
active_rule11 = np.fmax(temp_level_max, humidity_level_medium) # rule 11: max temp & medium humidity
active_rule12 = np.fmax(temp_level_max, humidity_level_high) # rule 12: max temp & high humidity



# intensity_activation_zero = fmin_multiple([active_rule6, active_rule3, intensity_zero], [])
# print(intensity_activation_zero)
print(f'{active_rule6=}')
print(f'{active_rule3=}')
print(f'{intensity_zero=}')
intensity_activation_zero = np.fmin(active_rule6, np.fmin(active_rule3, intensity_zero))
print(intensity_activation_zero)

intensity_activation_low = np.fmin(np.fmin(active_rule2, intensity_low), np.fmin(active_rule5, active_rule9))
intensity_activation_medium = np.fmin(np.fmin(active_rule1, intensity_medium), np.fmin(active_rule8, active_rule12))
intensity_activation_high = np.fmin( np.fmin(active_rule4, active_rule7), np.fmin(active_rule11, intensity_high))
intensity_activation_max = np.fmin(active_rule10, intensity_max)

intensity0 = np.zeros_like(x_intensity)

# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.fill_between(x_intensity, intensity0, intensity_activation_zero, facecolor='purple', alpha=0.7)
ax0.plot(x_intensity, intensity_zero, 'purple', linewidth=0.5, linestyle='--', )

ax0.fill_between(x_intensity, intensity0, intensity_activation_low, facecolor='b', alpha=0.7)
ax0.plot(x_intensity, intensity_low, 'b', linewidth=0.5, linestyle='--', )

ax0.fill_between(x_intensity, intensity0, intensity_activation_medium, facecolor='g', alpha=0.7)
ax0.plot(x_intensity, intensity_medium, 'g', linewidth=0.5, linestyle='--')

ax0.fill_between(x_intensity, intensity0, intensity_activation_high, facecolor='r', alpha=0.7)
ax0.plot(x_intensity, intensity_high, 'r', linewidth=0.5, linestyle='--')

ax0.fill_between(x_intensity, intensity0, intensity_activation_max, facecolor='orange', alpha=0.7)
ax0.plot(x_intensity, intensity_max, 'orange', linewidth=0.5, linestyle='--')

ax0.set_title('Output membership activity')

plt.tight_layout()
plt.show()

aggregated = np.fmax(np.fmax(intensity_activation_zero, intensity_activation_low), np.fmax(
                np.fmax(intensity_activation_medium, intensity_activation_high), intensity_activation_max))

intensity = fuzz.defuzz(x_intensity, aggregated, 'centroid')
intensity_activation = fuzz.interp_membership(x_intensity, aggregated, intensity)

# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.plot(x_intensity, intensity_zero, 'brown', linewidth=0.5, linestyle='--', )
ax0.plot(x_intensity, intensity_low, 'b', linewidth=0.5, linestyle='--', )
ax0.plot(x_intensity, intensity_medium, 'g', linewidth=0.5, linestyle='--')
ax0.plot(x_intensity, intensity_high, 'r', linewidth=0.5, linestyle='--')
ax0.plot(x_intensity, intensity_max, 'orange', linewidth=0.5, linestyle='--')
ax0.fill_between(x_intensity, intensity0, aggregated, facecolor='Orange', alpha=0.7)
ax0.plot([intensity, intensity], [0, intensity_activation], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('Aggregated membership and result (line)')

plt.tight_layout()

plt.show()

print(intensity)