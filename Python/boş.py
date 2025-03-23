import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [4, 5, 6])

# Sağ alt beyaz boşluğa yazı ekleyelim
plt.figtext(0.95, 0.01, 'Sağ Alt Dış Köşe',
            ha='right', va='bottom', fontsize=10, color='gray')

plt.show()
