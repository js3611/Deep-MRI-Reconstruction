from itertools import count

id_ctr = count()


def ensure_set_name(default_name, kwargs):
    """Ensure that the parameters contain names. Be careful, kwargs need to be
    passed as a dictionary here

    Parameters
    ----------
    default_name: string
        default name to set if neither name or pr is present, or if name is not
        present but pr is, the name becomes ``pr+default_name''
    kwargs: dict
        keyword arguments given to functions

    Returns
    -------
    kwargs: dict
    """
    if 'name' not in kwargs:
        raise Warning("You need to name the layers, "
                      "otherwise it simply won't work")
    global id_ctr
    if 'name' in kwargs and 'pr' in kwargs:
        kwargs['name'] = kwargs['pr']+kwargs['name']
    elif 'name' not in kwargs and 'pr' in kwargs:
        idx = next(id_ctr)
        kwargs['name'] = kwargs['pr'] + default_name + '_g' + str(idx)
    elif 'name' not in kwargs:
        idx = next(id_ctr)
        kwargs['name'] = default_name + '_g' + str(idx)
    return kwargs
