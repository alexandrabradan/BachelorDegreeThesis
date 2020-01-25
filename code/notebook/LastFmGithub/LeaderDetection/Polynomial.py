from itertools import zip_longest

__author__ = "https://www.python-course.eu/examples/polynomials.py"

class Polynomial:

    def __init__(self, *coefficients):
        """ input: coefficients are in the form a_n, ...a_1, a_0
        """
        self.coefficients = list(coefficients)  # tuple is turned into a list

    def __repr__(self):
        """
        method to return the canonical string representation
        of a polynomial.

        """
        return "Polynomial" + str(self.coefficients)

    def __call__(self, x):
        res = 0
        for coeff in self.coefficients:
            res = res * x + coeff
        return res

    def degree(self):
        return len(self.coefficients)

    def __add__(self, other):
        c1 = self.coefficients[::-1]
        c2 = other.coefficients[::-1]
        res = [sum(t) for t in zip_longest(c1, c2, fillvalue=0)]
        return Polynomial(*res)

    def __sub__(self, other):
        c1 = self.coefficients[::-1]
        c2 = other.coefficients[::-1]

        res = [t1 - t2 for t1, t2 in zip_longest(c1, c2, fillvalue=0)]
        return Polynomial(*res)

    def derivative(self):
        derived_coeffs = []
        exponent = len(self.coefficients) - 1
        for i in range(len(self.coefficients) - 1):
            derived_coeffs.append(self.coefficients[i] * exponent)
            exponent -= 1
        return Polynomial(*derived_coeffs)

    def __str__(self):
        res = ""
        degree = len(self.coefficients) - 1
        res += str(self.coefficients[0]) + "x^" + str(degree)
        for i in range(1, len(self.coefficients) - 1):
            coeff = self.coefficients[i]
            if coeff < 0:
                res += " - " + str(-coeff) + "x^" + str(degree - i)
            else:
                res += " + " + str(coeff) + "x^" + str(degree - i)

        if self.coefficients[-1] < 0:
            res += " - " + str(-self.coefficients[-1])
        else:
            res += " + " + str(self.coefficients[-1])

        return res

