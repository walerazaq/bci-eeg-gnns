# Defining readout function
def graph_readout(x, method, batch):
    if method == 'mean':
        return global_mean_pool(x,batch)

    elif method == 'meanmax':
        x_mean = global_mean_pool(x,batch)
        x_max = global_max_pool(x,batch)
        return torch.cat((x_mean, x_max), dim=1)

    elif method == 'sum':
        return global_add_pool(x,batch)

    else:
        raise ValueError('Undefined readout opertaion')
