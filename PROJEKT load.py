def load_data():
    path="Y-10_t.txt"
    with open(path) as f:
        lines = f.readlines()
        number_of_items = int(lines[0]) # pocet zariadeni
        print(number_of_items)
        widths = lines[1]
        #print(widths)
        widths = [int (x) for x in widths.split()] # sirka zariadeni "10 20 30" -> [10,20,30]
        tmp = lines[2:(2+number_of_items)]  # horny trojuholnik matice vzdialenosti od riadku 2 do 12, vsetky cisla
        data = []
        for line in tmp:
            row = []
            parts = line.split()
            for part in parts:
                row.append(float(part))
            data.append(row)

    #doplnenie dolnej casti matice podla symetrie
    n = len(data)
    for i in range(n):
        for j in range(i+1,n):
            data[j][i] = data[i][j]

    for row in data:
        print(row)
    return [widths, data]


widths, D = load_data()
#print(widths)
#print(D)