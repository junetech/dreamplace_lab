import datetime

from pyscipopt import Model, multidict, quicksum


def make_inst():
    """make_inst: prepare data for the diet model"""
    F, c, d = multidict(
        {  # cost # composition
            "QPounder": [
                1.84,
                {
                    "Cal": 510,
                    "Carbo": 34,
                    "Protein": 28,
                    "VitA": 15,
                    "VitC": 6,
                    "Calc": 30,
                    "Iron": 20,
                },
            ],
            "McLean": [
                2.19,
                {
                    "Cal": 370,
                    "Carbo": 35,
                    "Protein": 24,
                    "VitA": 15,
                    "VitC": 10,
                    "Calc": 20,
                    "Iron": 20,
                },
            ],
            "Big Mac": [
                1.84,
                {
                    "Cal": 500,
                    "Carbo": 42,
                    "Protein": 25,
                    "VitA": 6,
                    "VitC": 2,
                    "Calc": 25,
                    "Iron": 20,
                },
            ],
            "FFilet": [
                1.44,
                {
                    "Cal": 370,
                    "Carbo": 38,
                    "Protein": 14,
                    "VitA": 2,
                    "VitC": 0,
                    "Calc": 15,
                    "Iron": 10,
                },
            ],
            "Chicken": [
                2.29,
                {
                    "Cal": 400,
                    "Carbo": 42,
                    "Protein": 31,
                    "VitA": 8,
                    "VitC": 15,
                    "Calc": 15,
                    "Iron": 8,
                },
            ],
            "Fries": [
                0.77,
                {
                    "Cal": 220,
                    "Carbo": 26,
                    "Protein": 3,
                    "VitA": 0,
                    "VitC": 15,
                    "Calc": 0,
                    "Iron": 2,
                },
            ],
            "McMuffin": [
                1.29,
                {
                    "Cal": 345,
                    "Carbo": 27,
                    "Protein": 15,
                    "VitA": 4,
                    "VitC": 0,
                    "Calc": 20,
                    "Iron": 15,
                },
            ],
            "1% LFMilk": [
                0.60,
                {
                    "Cal": 110,
                    "Carbo": 12,
                    "Protein": 9,
                    "VitA": 10,
                    "VitC": 4,
                    "Calc": 30,
                    "Iron": 0,
                },
            ],
            "OrgJuice": [
                0.72,
                {
                    "Cal": 80,
                    "Carbo": 20,
                    "Protein": 1,
                    "VitA": 2,
                    "VitC": 120,
                    "Calc": 2,
                    "Iron": 2,
                },
            ],
        }
    )

    N, a, b = multidict(
        {  # min,max intake
            "Cal": [2000, None],
            "Carbo": [350, 375],
            "Protein": [55, None],
            "VitA": [100, None],
            "VitC": [100, None],
            "Calc": [100, None],
            "Iron": [100, None],
        }
    )

    return F, N, a, b, c, d


def make_model() -> Model:
    mdl = Model("Diet")

    # Parameters
    F, N, a, b, c, d = make_inst()

    # Create variables
    x, y, z = {}, {}, {}
    for j in F:
        x[j] = mdl.addVar(vtype="I", name="x(%s)" % j)
        y[j] = mdl.addVar(vtype="B", name="y(%s)" % j)
    for i in N:
        z[i] = mdl.addVar(lb=a[i], ub=b[i], name="z(%s)" % j)
    v = mdl.addVar(vtype="C", name="v")

    # Constraints:
    for i in N:
        mdl.addCons(quicksum(d[j][i] * x[j] for j in F) == z[i], name="Nutr(%s)" % i)

    mdl.addCons(quicksum(c[j] * x[j] for j in F) == v, name="Cost")

    for j in F:
        mdl.addCons(y[j] <= x[j], name="Eat(%s)" % j)

    # Objective:
    mdl.setObjective(quicksum(y[j] for j in F), "maximize")
    mdl.data = x, y, z, v

    return mdl


def get_solution(mdl: Model):
    print("Optimal value:", mdl.getObjVal())
    x, y, z, v = mdl.data
    for j in x:
        if mdl.getVal(x[j]) > 0:
            print(
                "{0:30s}: {1:3.1f} dishes --> {2:4.2f} added to objective".format(
                    j, mdl.getVal(x[j]), mdl.getVal(y[j])
                )
            )
    print("amount spent:", mdl.getObjVal())

    print("amount of nutrients:")
    for i in z:
        print("{0:30s}: {1:4.2f}".format(i, mdl.getVal(z[i])))


def main():
    s_dt = datetime.datetime.now()
    mdl = make_model()
    elapsed_d = datetime.datetime.now() - s_dt
    print(f"Math model building took {elapsed_d}"[:-3])
    s_dt = datetime.datetime.now()
    mdl.hideOutput()  # silent mode
    mdl.optimize()
    elapsed_d = datetime.datetime.now() - s_dt
    print(f"Math model solving took {elapsed_d}"[:-3])
    get_solution(mdl)


if __name__ == "__main__":
    START_DT = datetime.datetime.now()
    main()
    end_dt = datetime.datetime.now()
    elapsed_d = end_dt - START_DT
    print(f"{__name__} program end @ {end_dt}"[:-3] + f"; took total {elapsed_d}"[:-3])
