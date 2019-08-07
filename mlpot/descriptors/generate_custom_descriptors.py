import sympy as _sp
import re

rij, rik, costheta = _sp.symbols('rij rik costheta')

HEADER_TWO_BODY = """
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

HEADER_THREE_BODY = """
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

METHOD_TWO_BODY = """
double {}::{}(double rij)
{{
  return {};
}};
"""

EVAL_WITH_DERIVATIVES_TWO_BODY = """
void {}::eval_with_derivatives(double rij, double &G, double &dGdrij)
{{
  {};
}};
"""

METHOD_THREE_BODY = """
double {}::{}(double rij, double rik, double costheta)
{{
  return {};
}};
"""

DERIVATIVES_THREE_BODY = """
void {}::derivatives(double rij, double rik, double costheta,
  double &dGdrij, double &dGdrik, double &dGdcostheta)
{{
  {};
}};
"""

EVAL_WITH_DERIVATIVES_THREE_BODY = """
void {}::eval_with_derivatives(double rij, double rik, double costheta,
  double &G, double &dGdrij, double &dGdrik, double &dGdcostheta)
{{
  {};
}};
"""

CASE_STRING = """    case {}:
      symFun = std::make_shared<{}>(num_prms, prms, cutfun);
      break;
"""

SWITCH_STRING = """  if (strcmp(name, "{}") == 0)
  {{
    id = {};
  }}
"""

user_funs = {'fcut': 'cutfun->eval', 'dfcut': 'cutfun->derivative'}


def format_prms(num_prms, s):
    # Replace prm0 with prms[0]
    for i in range(num_prms):
        s = s.replace('prm{:d}'.format(i), 'prms[{:d}]'.format(i))
    return s


def format_py(s):
    return s


# Read custom symmetry function file
twoBodySymFuns = []
threeBodySymFuns = []
with open('custom_descriptors.txt', 'r') as fin:
    for line in fin:
        if line.startswith('TwoBodyDescriptor'):
            sp = line.split()
            twoBodySymFuns.append([sp[1], int(sp[2]), ' '.join(sp[3::])])
        if line.startswith('ThreeBodyDescriptor'):
            sp = line.split()
            threeBodySymFuns.append([sp[1], int(sp[2]), ' '.join(sp[3::])])

with open('symmetryFunctions.h', 'r') as fin:
    lines = fin.readlines()

CUSTOM_TWO_BODY_START = '// AUTOMATIC custom TwoBodyDescriptors start\n'
CUSTOM_TWO_BODY_END = '// AUTOMATIC custom TwoBodyDescriptors end\n'

CUSTOM_THREE_BODY_START = '// AUTOMATIC custom ThreeBodyDescriptors start\n'
CUSTOM_THREE_BODY_END = '// AUTOMATIC custom ThreeBodyDescriptors end\n'

lines = (lines[0:(lines.index(CUSTOM_TWO_BODY_START)+1)] +
         lines[lines.index(CUSTOM_TWO_BODY_END)::])
lines = (lines[0:(lines.index(CUSTOM_THREE_BODY_START)+1)] +
         lines[lines.index(CUSTOM_THREE_BODY_END)::])

with open('symmetryFunctions.h', 'w') as fout:
    for line in lines:
        fout.write(line)
        if line.startswith(CUSTOM_TWO_BODY_START):
            for symfun in twoBodySymFuns:
                fout.write(HEADER_TWO_BODY.format(symfun[0]))
        if line.startswith(CUSTOM_THREE_BODY_START):
            for symfun in threeBodySymFuns:
                fout.write(HEADER_THREE_BODY.format(symfun[0]))

with open('symmetryFunctions.cpp', 'r') as fin:
    lines = fin.readlines()

SWITCH_TWO_BODY_START = '// AUTOMATIC switch TwoBodyDescriptors start\n'
SWITCH_TWO_BODY_END = '// AUTOMATIC switch TwoBodyDescriptors end\n'

SWITCH_THREE_BODY_START = '// AUTOMATIC switch ThreeBodyDescriptors start\n'
SWITCH_THREE_BODY_END = '// AUTOMATIC switch ThreeBodyDescriptors end\n'

AVAILABLE_DESCRIPTORS_START = '// AUTOMATIC available_descriptors start\n'
AVAILABLE_DESCRIPTORS_end = '// AUTOMATIC available_descriptors end\n'

GET_TWO_BODY_START = '// AUTOMATIC get_two_body_descriptor start\n'
GET_TWO_BODY_END = '// AUTOMATIC get_two_body_descriptor end\n'

GET_THREE_BODY_START = '// AUTOMATIC get_three_body_descriptor start\n'
GET_THREE_BODY_END = '// AUTOMATIC get_three_body_descriptor end\n'

lines = (lines[0:(lines.index(CUSTOM_TWO_BODY_START)+1)] +
         lines[lines.index(CUSTOM_TWO_BODY_END)::])
lines = (lines[0:(lines.index(CUSTOM_THREE_BODY_START)+1)] +
         lines[lines.index(CUSTOM_THREE_BODY_END)::])
lines = (lines[0:(lines.index(SWITCH_TWO_BODY_START)+1)] +
         lines[lines.index(SWITCH_TWO_BODY_END)::])
lines = (lines[0:(lines.index(SWITCH_THREE_BODY_START)+1)] +
         lines[lines.index(SWITCH_THREE_BODY_END)::])
lines = (lines[0:(lines.index(AVAILABLE_DESCRIPTORS_START)+1)] +
         lines[lines.index(AVAILABLE_DESCRIPTORS_end)::])
lines = (lines[0:(lines.index(GET_TWO_BODY_START)+1)] +
         lines[lines.index(GET_TWO_BODY_END)::])
lines = (lines[0:(lines.index(GET_THREE_BODY_START)+1)] +
         lines[lines.index(GET_THREE_BODY_END)::])

with open('symmetryFunctions.cpp', 'w') as fout:
    for line in lines:
        fout.write(line)
        if line.startswith(CUSTOM_TWO_BODY_START):
            for symfun in twoBodySymFuns:
                parsed_symfun = _sp.sympify(symfun[2])
                fout.write(METHOD_TWO_BODY.format(symfun[0], 'eval',
                           format_prms(symfun[1], _sp.ccode(
                                symfun[2], user_functions=user_funs))))
                deriv = str(_sp.simplify(
                    _sp.Derivative(parsed_symfun, rij).doit()))
                deriv = deriv.replace(
                    'Derivative(fcut(rij), rij)', 'dfcut(rij)')
                fout.write(METHOD_TWO_BODY.format(
                    symfun[0], 'drij', format_prms(symfun[1], _sp.ccode(
                        deriv, user_functions=user_funs))))

                results = [_sp.simplify(parsed_symfun),
                           _sp.simplify(_sp.Derivative(
                                parsed_symfun, rij).doit())]

                simplified_results = [result.replace(
                    'Derivative(fcut(rij), rij)', 'dfcut(rij)') for result
                    in results]
                sub_exprs, simplified_results = _sp.cse(simplified_results)
                method_body = []
                for sub_expr in sub_exprs:
                    method_body.append(
                        'auto {} = {}'.format(
                            sub_expr[0], format_prms(symfun[1], _sp.ccode(
                                sub_expr[1], user_functions=user_funs))))
                method_body.append('G = {}'.format(format_prms(
                    symfun[1], _sp.ccode(
                        simplified_results[0], user_functions=user_funs))))
                method_body.append('dGdrij = {}'.format(format_prms(
                    symfun[1], _sp.ccode(
                        simplified_results[1], user_functions=user_funs))))

                fout.write(EVAL_WITH_DERIVATIVES_TWO_BODY.format(
                    symfun[0], ';\n  '.join(method_body)))
        elif line.startswith(CUSTOM_THREE_BODY_START):
            for symfun in threeBodySymFuns:
                parsed_symfun = _sp.sympify(symfun[2])
                fout.write(METHOD_THREE_BODY.format(
                    symfun[0], 'eval', format_prms(symfun[1], _sp.ccode(
                        symfun[2], user_functions=user_funs))))
                # Derivative with respect to rij
                deriv = _sp.simplify(
                    _sp.Derivative(parsed_symfun, rij).doit())
                deriv = _sp.sympify(re.sub(
                    'Derivative\(fcut\((?P<arg>.*?)\), (?P=arg)\)',  # NOQA
                    'dfcut(\g<arg>)', str(deriv))).doit()
                fout.write(METHOD_THREE_BODY.format(
                    symfun[0], 'drij', format_prms(symfun[1], _sp.ccode(
                        deriv, user_functions=user_funs))))
                # Derivative with respect to rik
                deriv = _sp.simplify(
                    _sp.Derivative(parsed_symfun, rik).doit())
                deriv = _sp.sympify(re.sub(
                    'Derivative\(fcut\((?P<arg>.*?)\), (?P=arg)\)',  # NOQA
                    'dfcut(\g<arg>)', str(deriv))).doit()
                fout.write(METHOD_THREE_BODY.format(
                    symfun[0], 'drik', format_prms(symfun[1], _sp.ccode(
                        deriv, user_functions=user_funs))))
                # Derivative with respect to costheta
                deriv = _sp.simplify(
                    _sp.Derivative(parsed_symfun, costheta).doit())
                deriv = _sp.sympify(re.sub(
                    'Derivative\(fcut\((?P<arg>.*?)\), (?P=arg)\)',  # NOQA
                    'dfcut(\g<arg>)', str(deriv))).doit()
                fout.write(METHOD_THREE_BODY.format(
                    symfun[0], 'dcostheta', format_prms(symfun[1], _sp.ccode(
                        deriv, user_functions=user_funs))))

                # Combined eval and derivatives
                results = [
                    _sp.simplify(parsed_symfun),
                    _sp.simplify(_sp.Derivative(parsed_symfun, rij).doit()),
                    _sp.simplify(_sp.Derivative(parsed_symfun, rik).doit()),
                    _sp.simplify(
                        _sp.Derivative(parsed_symfun, costheta).doit())]
                # Ugly work around to allow working with regex. Should maybe be
                # redone.
                results = [
                    _sp.sympify(re.sub(
                        'Derivative\(fcut\((?P<arg>.*?)\), (?P=arg)\)',  # NOQA
                        'dfcut(\g<arg>)', str(result))).doit()
                    for result in results]
                sub_exprs, simplified_results = _sp.cse(results)
                method_body = []
                for sub_expr in sub_exprs:
                    method_body.append(
                        'auto {} = {}'.format(
                            sub_expr[0], format_prms(symfun[1], _sp.ccode(
                                sub_expr[1], user_functions=user_funs))))
                method_body.append('G = {}'.format(format_prms(
                    symfun[1], _sp.ccode(
                        simplified_results[0], user_functions=user_funs))))
                method_body.append('dGdrij = {}'.format(format_prms(
                    symfun[1], _sp.ccode(
                        simplified_results[1], user_functions=user_funs))))
                method_body.append('dGdrik = {}'.format(format_prms(
                    symfun[1], _sp.ccode(
                        simplified_results[2], user_functions=user_funs))))
                method_body.append('dGdcostheta = {}'.format(
                    format_prms(symfun[1], _sp.ccode(
                        simplified_results[3], user_functions=user_funs))))

                fout.write(EVAL_WITH_DERIVATIVES_THREE_BODY.format(
                    symfun[0], ';\n  '.join(method_body)))

                # Derivatives with respect to the three arguments
                sub_exprs, simplified_derivs = _sp.cse(results[1::])
                method_body = []
                for sub_expr in sub_exprs:
                    method_body.append(
                        'auto {} = {}'.format(
                            sub_expr[0], format_prms(symfun[1], _sp.ccode(
                                sub_expr[1], user_functions=user_funs))))
                method_body.append('dGdrij = {}'.format(format_prms(
                    symfun[1], _sp.ccode(
                        simplified_derivs[0], user_functions=user_funs))))
                method_body.append('dGdrik = {}'.format(format_prms(
                    symfun[1], _sp.ccode(
                        simplified_derivs[1], user_functions=user_funs))))
                method_body.append('dGdcostheta = {}'.format(
                    format_prms(symfun[1], _sp.ccode(
                        simplified_derivs[2], user_functions=user_funs))))

                fout.write(DERIVATIVES_THREE_BODY.format(
                    symfun[0], ';\n  '.join(method_body)))
        elif line.startswith(AVAILABLE_DESCRIPTORS_START):
            fout.write('  printf("TwoBodySymmetryFunctions:'
                       ' (key: name, # of parameters)\\n");\n')
            for i, symfun in enumerate(twoBodySymFuns):
                fout.write('  printf("{}: {}, {}\\n");\n'.format(
                    i, symfun[0], symfun[1]))
            fout.write('  printf("ThreeBodySymmetryFunctions:'
                       ' (key: name, # of parameters)\\n");\n')
            for i, symfun in enumerate(threeBodySymFuns):
                fout.write('  printf("{}: {}, {}\\n");\n'.format(
                    i, symfun[0], symfun[1]))

        elif line.startswith(SWITCH_TWO_BODY_START):
            for i, symfun in enumerate(twoBodySymFuns):
                fout.write(CASE_STRING.format(i, symfun[0]))
        elif line.startswith(SWITCH_THREE_BODY_START):
            for i, symfun in enumerate(threeBodySymFuns):
                fout.write(CASE_STRING.format(i, symfun[0]))
        elif line.startswith(GET_TWO_BODY_START):
            for i, symfun in enumerate(twoBodySymFuns):
                fout.write(SWITCH_STRING.format(symfun[0], i))
        elif line.startswith(GET_THREE_BODY_START):
            for i, symfun in enumerate(threeBodySymFuns):
                fout.write(SWITCH_STRING.format(symfun[0], i))
