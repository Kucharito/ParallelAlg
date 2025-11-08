def load_data():
    path = "Y-10_t.txt"
    with open(path) as f:
        lines = f.readlines()

        number_of_items = int(lines[0])          # první řádek = počet zařízení
        widths = [int(x) for x in lines[1].split()]  # druhý řádek = šířky zařízení

        tmp = lines[2:(2 + number_of_items)]     # další řádky = horní trojúhelník matice
        data = []
        for line in tmp:
            parts = line.split()
            row = [float(part) for part in parts]
            data.append(row)

    # doplnění dolní části matice podle symetrie
    n = len(data)
    for i in range(n+1):
        for j in range(i + 1, n):
            data[j][i] = data[i][j]

    return [widths, data]
