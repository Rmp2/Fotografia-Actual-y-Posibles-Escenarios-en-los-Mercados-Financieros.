'''En este archivo pondré todas mis funciones de carga de datos
a la base de datos.'''
import csv

def writelstcsv(guest_list, filename):
    """Write the list to csv file."""

    with open(filename, "w", encoding='utf-8') as outfile:
        for entries in guest_list:
            outfile.write(entries)
            outfile.write("\n")

def opencsv(filename, lst):
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            lst.append(', '.join(row))
    return lst
