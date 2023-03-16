import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages

 # 1
def score(methods):
    return df[methods].values.tolist()


df = pd.read_csv('plot.csv', index_col=0)
x = df.index.tolist()
y = score('NB')
pdf = PdfPages('plot.pdf')
plt.figure(figsize=(12, 12))
l3 = plt.plot(x, y, 'r', label='NB')
x_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.title('IFS Curve', fontsize=30)
plt.xlabel('Number of Transcripts', fontsize=20)
plt.ylabel('AUC', fontsize=20)
plt.legend(loc='best', fontsize=15)
plt.ylim(0.95, 1.001)
pdf.savefig()
plt.close()
pdf.close()
