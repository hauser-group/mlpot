import sympy as _sp
import re

rij, rik, costheta = _sp.symbols("rij rik costheta")

header_twoBody = """
class {0}: public TwoBodySymmetryFunction
{{
    public:
        {0}(int num_prms, double* prms,
          std::shared_ptr<CutoffFunction> cutfun):
          TwoBodySymmetryFunction(num_prms, prms, cutfun){{}};
        double eval(double rij);
        double drij(double rij);
        void eval_with_derivatives(double rij, double &G, double &dGdrij);
}};
"""

header_threeBody = """
class {0}: public ThreeBodySymmetryFunction
{{
  public:
    {0}(int num_prms, double* prms,
      std::shared_ptr<CutoffFunction> cutfun):
      ThreeBodySymmetryFunction(num_prms, prms, cutfun){{}};
    double eval(double rij, double rik, double costheta);
    double drij(double rij, double rik, double costheta);
    double drik(double rij, double rik, double costheta);
    double dcostheta(double rij, double rik, double costheta);
    void derivatives(double rij, double rik, double costheta,
      double &dGdrij, double &dGdrik, double &dGdcostheta);
    void eval_with_derivatives(double rij, double rik, double costheta,
      double &G, double &dGdrij, double &dGdrik, double &dGdcostheta);
}};
"""

method_twoBody = """
double {}::{}(double rij)
{{
  return {};
}};
"""

eval_with_derivatives_twoBody = """
void {}::eval_with_derivatives(double rij, double &G, double &dGdrij)
{{
  {};
}};
"""

method_threeBody = """
double {}::{}(double rij, double rik, double costheta)
{{
  return {};
}};
"""

derivatives_threeBody = """
void {}::derivatives(double rij, double rik, double costheta,
  double &dGdrij, double &dGdrik, double &dGdcostheta)
{{
  {};
}};
"""

eval_with_derivatives_threeBody = """
void {}::eval_with_derivatives(double rij, double rik, double costheta,
  double &G, double &dGdrij, double &dGdrik, double &dGdcostheta)
{{
  {};
}};
"""

case_string = """    case {}:
      symFun = std::make_shared<{}>(num_prms, prms, cutfun);
      break;
"""

switch_string = """  if (strcmp(name, "{}") == 0)
  {{
    id = {};
  }}
"""

user_funs = {"fcut":"cutfun->eval", "dfcut":"cutfun->derivative"}

def format_prms(num_prms, s):
    # Replace prm0 with prms[0]
    for i in range(num_prms):
        s = s.replace("prm{:d}".format(i), "prms[{:d}]".format(i))
    return s

def format_py(s):
    return s

# Read custom symmetry function file
twoBodySymFuns = []
threeBodySymFuns = []
with open("customSymFuns.txt", "r") as fin:
    for line in fin:
        if line.startswith("TwoBodySymFun"):
            sp = line.split()
            twoBodySymFuns.append([sp[1], int(sp[2]), " ".join(sp[3::])])
        if line.startswith("ThreeBodySymFun"):
            sp = line.split()
            threeBodySymFuns.append([sp[1], int(sp[2]), " ".join(sp[3::])])

with open("symmetryFunctions.h", "r") as fin:
    lines = fin.readlines()

lines = (lines[0:(lines.index("// AUTOMATIC Start of custom TwoBodySymFuns\n")+1)] +
    lines[lines.index("// AUTOMATIC End of custom TwoBodySymFuns\n")::])
lines = (lines[0:(lines.index("// AUTOMATIC Start of custom ThreeBodySymFuns\n")+1)] +
    lines[lines.index("// AUTOMATIC End of custom ThreeBodySymFuns\n")::])

with open("symmetryFunctions.h", "w") as fout:
    for line in lines:
        fout.write(line)
        if line.startswith("// AUTOMATIC Start of custom TwoBodySymFuns"):
            for symfun in twoBodySymFuns:
                fout.write(header_twoBody.format(symfun[0]))
        if line.startswith("// AUTOMATIC Start of custom ThreeBodySymFuns"):
            for symfun in threeBodySymFuns:
                fout.write(header_threeBody.format(symfun[0]))

with open("symmetryFunctions.cpp", "r") as fin:
    lines = fin.readlines()

lines = (lines[0:(lines.index("// AUTOMATIC Start of custom TwoBodySymFuns\n")+1)] +
    lines[lines.index("// AUTOMATIC End of custom TwoBodySymFuns\n")::])
lines = (lines[0:(lines.index("// AUTOMATIC Start of custom ThreeBodySymFuns\n")+1)] +
    lines[lines.index("// AUTOMATIC End of custom ThreeBodySymFuns\n")::])
lines = (lines[0:(lines.index("// AUTOMATIC TwoBodySymmetryFunction switch start\n")+1)] +
    lines[lines.index("// AUTOMATIC TwoBodySymmetryFunction switch end\n")::])
lines = (lines[0:(lines.index("// AUTOMATIC ThreeBodySymmetryFunction switch start\n")+1)] +
    lines[lines.index("// AUTOMATIC ThreeBodySymmetryFunction switch end\n")::])
lines = (lines[0:(lines.index("// AUTOMATIC available_symFuns start\n")+1)] +
    lines[lines.index("// AUTOMATIC available_symFuns end\n")::])
lines = (lines[0:(lines.index("// AUTOMATIC get_TwoBodySymFuns start\n")+1)] +
    lines[lines.index("// AUTOMATIC get_TwoBodySymFuns end\n")::])
lines = (lines[0:(lines.index("// AUTOMATIC get_ThreeBodySymFuns start\n")+1)] +
    lines[lines.index("// AUTOMATIC get_ThreeBodySymFuns end\n")::])

with open("symmetryFunctions.cpp", "w") as fout:
    for line in lines:
        fout.write(line)
        if line.startswith("// AUTOMATIC Start of custom TwoBodySymFuns"):
            for symfun in twoBodySymFuns:
                parsed_symfun = _sp.sympify(symfun[2])
                fout.write(method_twoBody.format(symfun[0],"eval",
                    format_prms(symfun[1],_sp.ccode(symfun[2],
                    user_functions = user_funs))))
                deriv = str(_sp.simplify(
                    _sp.Derivative(parsed_symfun, rij).doit()))
                deriv = deriv.replace("Derivative(fcut(rij), rij)", "dfcut(rij)")
                fout.write(method_twoBody.format(symfun[0],"drij",
                    format_prms(symfun[1],_sp.ccode(deriv,
                    user_functions = user_funs))))

                results = [_sp.simplify(parsed_symfun),
                    _sp.simplify(_sp.Derivative(parsed_symfun, rij).doit())]

                simplified_results = [result.replace(
                    "Derivative(fcut(rij), rij)", "dfcut(rij)") for result
                    in results]
                sub_exprs, simplified_results = _sp.cse(simplified_results)
                method_body = []
                for sub_expr in sub_exprs:
                    method_body.append(
                        "auto {} = {}".format(
                        sub_expr[0], format_prms(symfun[1],
                        _sp.ccode(sub_expr[1], user_functions = user_funs))))
                method_body.append("G = {}".format(format_prms(symfun[1],
                    _sp.ccode(simplified_results[0],
                    user_functions = user_funs))))
                method_body.append("dGdrij = {}".format(format_prms(symfun[1],
                    _sp.ccode(simplified_results[1],
                    user_functions = user_funs))))

                fout.write(eval_with_derivatives_twoBody.format(symfun[0],
                    ";\n  ".join(method_body)))
        elif line.startswith("// AUTOMATIC Start of custom ThreeBodySymFuns"):
            for symfun in threeBodySymFuns:
                parsed_symfun = _sp.sympify(symfun[2])
                fout.write(method_threeBody.format(symfun[0],"eval",
                    format_prms(symfun[1],_sp.ccode(symfun[2],
                    user_functions = user_funs))))
                # Derivative with respect to rij
                deriv = _sp.simplify(
                    _sp.Derivative(parsed_symfun, rij).doit())
                deriv = _sp.sympify(re.sub(
                    "Derivative\(fcut\((?P<arg>.*?)\), (?P=arg)\)",
                    "dfcut(\g<arg>)", str(deriv))).doit()
                fout.write(method_threeBody.format(symfun[0],"drij",
                    format_prms(symfun[1],_sp.ccode(deriv,
                    user_functions = user_funs))))
                # Derivative with respect to rik
                deriv = _sp.simplify(
                    _sp.Derivative(parsed_symfun, rik).doit())
                deriv = _sp.sympify(re.sub(
                    "Derivative\(fcut\((?P<arg>.*?)\), (?P=arg)\)",
                    "dfcut(\g<arg>)", str(deriv))).doit()
                fout.write(method_threeBody.format(symfun[0],"drik",
                    format_prms(symfun[1],_sp.ccode(deriv,
                    user_functions = user_funs))))
                # Derivative with respect to costheta
                deriv = _sp.simplify(
                    _sp.Derivative(parsed_symfun, costheta).doit())
                deriv = _sp.sympify(re.sub(
                    "Derivative\(fcut\((?P<arg>.*?)\), (?P=arg)\)",
                    "dfcut(\g<arg>)", str(deriv))).doit()
                fout.write(method_threeBody.format(symfun[0],"dcostheta",
                    format_prms(symfun[1],_sp.ccode(deriv,
                    user_functions = user_funs))))

                # Combined eval and derivatives
                results = [_sp.simplify(parsed_symfun),
                    _sp.simplify(_sp.Derivative(parsed_symfun, rij).doit()),
                    _sp.simplify(_sp.Derivative(parsed_symfun, rik).doit()),
                    _sp.simplify(_sp.Derivative(parsed_symfun, costheta).doit())]
                # Ugly work around to allow working with regex. Should maybe be
                # redone.
                results = [_sp.sympify(re.sub(
                    "Derivative\(fcut\((?P<arg>.*?)\), (?P=arg)\)",
                    "dfcut(\g<arg>)", str(result))).doit() for result in results]
                sub_exprs, simplified_results = _sp.cse(results)
                method_body = []
                for sub_expr in sub_exprs:
                    method_body.append(
                        "auto {} = {}".format(
                        sub_expr[0], format_prms(symfun[1],
                        _sp.ccode(sub_expr[1], user_functions = user_funs))))
                method_body.append("G = {}".format(format_prms(symfun[1],
                    _sp.ccode(simplified_results[0],
                    user_functions = user_funs))))
                method_body.append("dGdrij = {}".format(format_prms(symfun[1],
                    _sp.ccode(simplified_results[1],
                    user_functions = user_funs))))
                method_body.append("dGdrik = {}".format(format_prms(symfun[1],
                    _sp.ccode(simplified_results[2],
                    user_functions = user_funs))))
                method_body.append("dGdcostheta = {}".format(
                    format_prms(symfun[1], _sp.ccode(simplified_results[3],
                    user_functions = user_funs))))

                fout.write(eval_with_derivatives_threeBody.format(symfun[0],
                    ";\n  ".join(method_body)))

                # Derivatives with respect to the three arguments
                sub_exprs, simplified_derivs = _sp.cse(results[1::])
                method_body = []
                for sub_expr in sub_exprs:
                    method_body.append(
                        "auto {} = {}".format(
                        sub_expr[0], format_prms(symfun[1],
                        _sp.ccode(sub_expr[1], user_functions = user_funs))))
                method_body.append("dGdrij = {}".format(format_prms(symfun[1],
                    _sp.ccode(simplified_derivs[0],
                    user_functions = user_funs))))
                method_body.append("dGdrik = {}".format(format_prms(symfun[1],
                    _sp.ccode(simplified_derivs[1],
                    user_functions = user_funs))))
                method_body.append("dGdcostheta = {}".format(
                    format_prms(symfun[1], _sp.ccode(simplified_derivs[2],
                    user_functions = user_funs))))

                fout.write(derivatives_threeBody.format(symfun[0],
                    ";\n  ".join(method_body)))
        elif line.startswith("// AUTOMATIC available_symFuns start"):
            fout.write('  printf("TwoBodySymmetryFunctions: (key: name, # of parameters)\\n");\n')
            for i, symfun in enumerate(twoBodySymFuns):
                fout.write('  printf("{}: {}, {}\\n");\n'.format(i, symfun[0], symfun[1]))
            fout.write('  printf("ThreeBodySymmetryFunctions: (key: name, # of parameters)\\n");\n')
            for i, symfun in enumerate(threeBodySymFuns):
                fout.write('  printf("{}: {}, {}\\n");\n'.format(i, symfun[0], symfun[1]))

        elif line.startswith("// AUTOMATIC TwoBodySymmetryFunction switch start"):
            for i, symfun in enumerate(twoBodySymFuns):
                fout.write(case_string.format(i, symfun[0]))
        elif line.startswith("// AUTOMATIC ThreeBodySymmetryFunction switch start"):
            for i, symfun in enumerate(threeBodySymFuns):
                fout.write(case_string.format(i, symfun[0]))
        elif line.startswith("// AUTOMATIC get_TwoBodySymFuns start"):
            for i, symfun in enumerate(twoBodySymFuns):
                fout.write(switch_string.format(symfun[0], i))
        elif line.startswith("// AUTOMATIC get_ThreeBodySymFuns start"):
            for i, symfun in enumerate(threeBodySymFuns):
                fout.write(switch_string.format(symfun[0], i))
