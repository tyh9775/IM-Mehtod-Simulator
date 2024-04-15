import matplotlib.pyplot as plt

data=[[1,[1,1,2,1]],[2,[2,4,3,4]],[4,[3,3,1,5]]]

plt.figure(figsize=(10, 6))

for label, values in data:
    plt.hist(values, alpha=0.5, label=f'Label {label}')

plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Multiple Histograms')
plt.legend()
plt.grid(True)
plt.show()
