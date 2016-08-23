"""
Functions for dealing with curves, I suppose?

@jonathanronen
"""

def argelbow(curve, index=None, plot=False):
    """
    Finds the elbow of a curve (for instance, singular values) using "max distance from straight line".
    Basically: Draw a straight line from the start to the end of the curve. For each point on the curve, calculate the distance
    between the curve and the closest point on the straight line. The point on the curve with the farthest nearest point on the straight line
    is the elbow.
    """
    index = index or np.arange(len(curve))

    pt_A = np.array([index[0], curve[0]])
    pt_B = np.array([index[-1], curve[-1]])

    r = pt_B - pt_A
    rnorm = r/(np.sqrt(r.dot(r)))

    if plot:
        plt.figure()
        plt.plot(index, curve, label='curve')
        plt.hold(True)
        plt.plot([index[0], index[-1]] , [curve[0], curve[-1]], '--k')

    perplines = list()
    distances2 = list()
    for pt_x, pt_y in list(zip(index, curve))[1:-1]:
        pv = np.array([pt_x - pt_A[0], pt_y - pt_A[1]])
        pvprojr = np.dot(pv, rnorm) * rnorm
        p_on_line = pt_A + pvprojr
        perplines.append(([p_on_line[0], pt_x], [p_on_line[1], pt_y]))
        distances2.append(np.square(p_on_line[0]-pt_x) + np.square(p_on_line[1]-pt_y))
    maxd = np.argmax(distances2)
    if plot:
        for i, (x, y) in enumerate(perplines):
            color = 'gold' if i==maxd else 'grey'
            plt.plot(x, y, linestyle='--', color=color)
            if i==maxd:
                plt.plot([x[1]], [y[1]], 'o', color='red', label='elbow')

        plt.title('Max min distance to straight line elbow method')
        plt.legend()
    return maxd
