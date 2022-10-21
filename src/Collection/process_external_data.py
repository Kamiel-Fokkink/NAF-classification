import pandas as pd
import re


def main():
    naf_file = 'NAF_2020.txt'
    df = pd.DataFrame({'naf':[], 'text':[]})
    f = open(naf_file, 'r')
    Lines = f.readlines()
    
    i = 1
    current_line = ''
    within_line = False
    for l in Lines:
        line = l.strip()
        naf = '0' + str(i) if len(str(i))==1 else str(i)
        new_i = i + 1 if i not in [3, 13, 23, 33, 39, 43, 47, 53, 56, 66, 75, 82, 88] else i + 2
        new_naf = '0' + str(new_i) if len(str(new_i))==1 else str(new_i)

        # deleting header and footer 
        if re.findall(r'Nomenclature d’Activités Française NAF', line) or re.findall(r'Classification des Produits', line):
            continue
        
        # deleting the part after "cette division ne comprend pas" and "produits associes" and "NC"
        if re.findall(r'ne comprend pas', line) or re.findall(r'Produits asso', line) or re.findall(r'NC :', line):
            within_line = False
            continue 

        if naf + '.' == line[:3]:
            if current_line:
                df = df.append({'naf': naf, 'text': current_line}, ignore_index=True)
            current_line = line.lstrip('0123456789.-,p ').replace('CC :', '').replace('CA :', '').replace('Z ', '')
            within_line = True
        elif new_naf + ' ' == line[:3]:
            if current_line:
                df = df.append({'naf': naf, 'text':current_line}, ignore_index=True)
            i = new_i
            current_line = line.lstrip('0123456789.-,p ').replace('CC :', '').replace('CA :', '').replace('Z ', '')
            within_line = True
        elif naf + ' ' == line[:3] or within_line:
            current_line += ' ' + line.lstrip('0123456789.-,p ').replace('CC :', '').replace('CA :', '').replace('Z ', '')
        else:
            continue
    
    df.to_csv('naf_2020.csv')

if __name__ == '__main__' :
    main()
