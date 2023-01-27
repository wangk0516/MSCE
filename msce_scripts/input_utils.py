

# -----------------------------------------------------------------------------

def read_CE_params(filename = 'CE_Params.in'):
    CEOptions = {}

    params_file = open(filename, 'r')
    params_info = params_file.readlines()
    params_file.close()

    for line in params_info:
        content = line.replace('=', '').split()
        if content[0] == 'K-Fold':
            CEOptions[content[0]] = int(content[1])
        elif content[0] == 'EPS_LENGTH':
            CEOptions[content[0]] = float(content[1])
        elif content[0] == 'Prediction':
            CEOptions[content[0]] = eval(content[1])
        elif content[0] == 'MaxError':
            CEOptions[content[0]] = float(content[1])
        elif content[0] == 'GSError':
            CEOptions[content[0]] = float(content[1])
        elif content[0] == 'Cut-Off':
            cut_off = []
            for K in range(1, len(content)):
                cut_off.append(float(content[K]))
            # end for-K
            CEOptions[content[0]] = cut_off
        else:
            CEOptions[content[0]] = content[1]
        # end if
    # end for-line

    return CEOptions

# -----------------------------------------------------------------------------

