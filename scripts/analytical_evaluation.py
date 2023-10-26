import sympy as sym


def main():
    dt = sym.Symbol("dt")
    dfdt = sym.Symbol("dfdt")

    D = sym.Symbol("D")
    M = sym.Symbol("M")

    ll = sym.Symbol("l")

    A0 = sym.Matrix([[1, dt], [dt * D / M * dfdt, 1 - dt * D / M]])
    det = (A0[0, 0] - ll) * (A0[1, 1] - ll) - A0[0, 1] * A0[1, 0]

    print(sym.expand(det))


if (__name__) == "__main__":
    main()
