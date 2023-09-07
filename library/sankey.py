import plotly.graph_objects as go


def _code_mapping(df, src, targ):
    """ Maps labels / strings in src and target
    and converts them to integers 0,1,2,3... """

    # Extract distinct labels
    labels = sorted(list(set(list(df[src]) + list(df[targ]))))

    # define integer codes
    codes = list(range(len(labels)))

    # pair labels with list
    lc_map = dict(zip(labels, codes))

    # in df, substitute codes for labels
    df = df.replace({src: lc_map, targ: lc_map})

    return df, labels


def make_sankey(df, src, targ, vals=None, **kwargs):
    """Generate the sankey diagram """

    df, labels = _code_mapping(df, src, targ)

    if vals:
        values = df[vals]
    else:
        values = [1] * len(df)

    pad = kwargs.get('pad', 50)
    thickness = kwargs.get('thickness', 30)
    line_color = kwargs.get('line_color', 'black')
    line_width = kwargs.get('line_width', 1)

    link = {'source': df[src], 'target': df[targ], 'value': values}
    node = {'label': labels, 'pad': pad, 'thickness': thickness,
            'line': {'color': line_color, 'width': line_width}}

    sk = go.Sankey(link=link, node=node)
    fig = go.Figure(sk)
    fig.show()
