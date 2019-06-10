'''
In order to numerically solve a ordinary differential equation, we
will simply use the classical 4th order Runge-Kutta method, based
on the Wikipedia article on the subject:
http://en.wikipedia.org/wiki/Runge-Kutta_methods
Here we will allow the ODE to be specified as a python function,
    f = lambda t, y: ...
representing dy/dt, and an initial condition y0 = y(t0).
The code will be written mainly with clarity in mind, rather than
striving for maximal efficiency or generality.
'''

# Some basic functional programming libraries
from functools import reduce
from operator import mul

def runge_kutta(q, f, t0, q0, y0, h):
    '''
    Classical Runge-Kutta method for dy/dt = f(t, y), y(t0) = y0,
    with step h, and the specified tolerance and max_steps.
    This function is a generator which can give infinitely many points
    of the estimated solution, in pairs (t[n], y[n]).
    To get only finitely many values of
    the solution we can for example do,
        >>> from itertools import islice
        >>> list(islice(runge_kutta(f, t0, h), n))
        [(t[0], y[0]), (t[1], y[1]), ..., (t[n], y[n])]
    and infact, we could define another function to do this like,
        >>> runge_kutta_N = lambda f, t0, y0, h, N: list(islice(
        ...     runge_kutta(f, t0, y0, h), N))
    It would also be easy to change this function to take an extra
    parameter N and then return a list of the first N, (t_n, y_n),
    directly (by replacing the while loop with for n in range(N)).
    Note also that whilst the question asks for a solution, this
    function only returns an approximation of the solution at
    certain points. We can turn use this to generate a continuous
    function passing through the points specified using either of
    interpolation methods specified lower down the file.
    '''
    # y and t represent y[n] and t[n] respectively at each stage
    y = y0
    q = q0
    t = t0

    # Whilst it would be more elegant to write this recursively,
    # in Python this would be very inefficient, and lead to errors when
    # many iterations are required, as the language does not perform
    # tail call optimisations as would be the case in languages such
    # as C, Lisp, or Haskell.
    #
    # Instead we use a simple infinite loop, which will yield more values
    # of the function indefinitely.
    while True:
        # Generate the next values of the solution y
        yield t, q, y

        # Values for weighted average (compare with Wikipedia)
        kq1 = q(t, y)
        kq2 = q(t + h/2, y + (h/2)*kq1)
        kq3 = q(t + h/2, y + (h/2)*kq2)
        kq4 = q(t + h, y + h*kq3)
        ky1 = f(t, q)
        ky2 = f(t + h/2, q + (h/2)*ky1)
        ky3 = f(t + h/2, q + (h/2)*ky2)
        ky4 = f(t + h, q + h*ky3)

        # Calculate the new value of y and t as per the Runge-Kutta method
        q = q + (h/6)*(kq1 + 2*kq2 + 2*kq3 + kq4)
        y = y + (h/6)*(ky1 + 2*ky2 + 2*ky3 + ky4)
        t = t + h

if __name__ == '__main__':
    # A demonstration of the method for,
    # dy/dt = (1/3)*y + e^t*y^4, y(0) = 1
    # (chosen to test the method, as an explict solution does exist)
    print('dy/dt = (1/3)*y + e^t*y^4, y(0) = 1')
    from itertools import islice
    from math import exp
    # Evaluate the solution at some points
    ty1s = list(islice(runge_kutta(
        lambda q, y: q,
        lambda t, q:  5*q, # f(t, y)
        0., # t0
        1., # y0
        0.1,
    ), 10))
    # Print some values
    print(ty1s)

    print('\n'.join('y({:.3f}) = {}'.format(t, y) for t, y in ty1s))

