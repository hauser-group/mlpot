import sympy as _sp
import re

rij, rik, costheta = _sp.symbols('rij rik costheta')

HEADER_TWO_BODY = """
class {0}: public TwoBodyDescriptor
{{
    public:
        {0}(int num_prms, double* prms,
          std::shared_ptr<CutoffFunction> cutfun):
          TwoBodyDescriptor(num_prms, prms, cutfun){{}};
        double eval(double rij);
        double drij(double rij);
        void eval_with_derivatives(double rij, double &G, double &dGdrij);
}};
"""

HEADER_THREE_BODY = """
class {0}: public ThreeBodyDescriptor
{{
  public:
    {0}(int num_prms, double* prms,
      std::shared_ptr<CutoffFunction> cutfun):
      ThreeBodyDescriptor(num_prms, prms, cutfun){{}};
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

IF_FUN_ID = ('  if (funtype == {}) '
             + 'symFun = std::make_shared<{}>(num_prms, prms, cutfun);\n')
ELIF_FUN_ID = ('  else if (funtype == {}) '
               + 'symFun = std::make_shared<{}>(num_prms, prms, cutfun);\n')

IF_STRING = '  if (strcmp(name, "{}") == 0) id = {};\n'
ELIF_STRING = '  else if (strcmp(name, "{}") == 0) id = {};\n'

user_funs = {'fcut': 'cutfun->eval', 'dfcut': 'cutfun->derivative'}


def format_prms(num_prms, s):
    # Replace prm0 with prms[0]
    for i in range(num_prms):
        s = s.replace('prm{:d}'.format(i), 'prms[{:d}]'.format(i))
    return s


def format_py(s):
    return s


class Descriptor():

    def __init__(self, name, num_prms, expr):
        self.name = name
        self.num_prms = num_prms
        self.expr = expr


# Read custom descriptor file
two_body_descriptors = []
three_body_descriptors = []
with open('custom_descriptors.txt', 'r') as fin:
    for line in fin:
        if line.startswith('TwoBodyDescriptor'):
            sp = line.split()
            two_body_descriptors.append(
                Descriptor(sp[1], int(sp[2]), ' '.join(sp[3::])))
        if line.startswith('ThreeBodyDescriptor'):
            sp = line.split()
            three_body_descriptors.append(
                Descriptor(sp[1], int(sp[2]), ' '.join(sp[3::])))

with open('descriptors.h', 'r') as fin:
    lines = fin.readlines()

CUSTOM_TWO_BODY_START = '// AUTOMATIC custom TwoBodyDescriptors start\n'
CUSTOM_TWO_BODY_END = '// AUTOMATIC custom TwoBodyDescriptors end\n'

CUSTOM_THREE_BODY_START = '// AUTOMATIC custom ThreeBodyDescriptors start\n'
CUSTOM_THREE_BODY_END = '// AUTOMATIC custom ThreeBodyDescriptors end\n'

lines = (lines[0:(lines.index(CUSTOM_TWO_BODY_START)+1)] +
         lines[lines.index(CUSTOM_TWO_BODY_END)::])
lines = (lines[0:(lines.index(CUSTOM_THREE_BODY_START)+1)] +
         lines[lines.index(CUSTOM_THREE_BODY_END)::])

with open('descriptors.h', 'w') as fout:
    for line in lines:
        fout.write(line)
        if line.startswith(CUSTOM_TWO_BODY_START):
            for descriptor in two_body_descriptors:
                fout.write(HEADER_TWO_BODY.format(descriptor.name))
        if line.startswith(CUSTOM_THREE_BODY_START):
            for descriptor in three_body_descriptors:
                fout.write(HEADER_THREE_BODY.format(descriptor.name))

with open('descriptors.cpp', 'r') as fin:
    lines = fin.readlines()

SWITCH_TWO_BODY_START = '// AUTOMATIC switch TwoBodyDescriptors start\n'
SWITCH_TWO_BODY_END = '// AUTOMATIC switch TwoBodyDescriptors end\n'

SWITCH_THREE_BODY_START = '// AUTOMATIC switch ThreeBodyDescriptors start\n'
SWITCH_THREE_BODY_END = '// AUTOMATIC switch ThreeBodyDescriptors end\n'

AVAILABLE_TWO_BODY_START = (
    '// AUTOMATIC available_two_body_descriptors start\n')
AVAILABLE_TWO_BODY_END = '// AUTOMATIC available_two_body_descriptors end\n'
AVAILABLE_THREE_BODY_START = (
    '// AUTOMATIC available_three_body_descriptors start\n')
AVAILABLE_THREE_BODY_END = (
    '// AUTOMATIC available_three_body_descriptors end\n')

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
lines = (lines[0:(lines.index(AVAILABLE_TWO_BODY_START)+1)] +
         lines[lines.index(AVAILABLE_TWO_BODY_END)::])
lines = (lines[0:(lines.index(AVAILABLE_THREE_BODY_START)+1)] +
         lines[lines.index(AVAILABLE_THREE_BODY_END)::])
lines = (lines[0:(lines.index(GET_TWO_BODY_START)+1)] +
         lines[lines.index(GET_TWO_BODY_END)::])
lines = (lines[0:(lines.index(GET_THREE_BODY_START)+1)] +
         lines[lines.index(GET_THREE_BODY_END)::])

with open('descriptors.cpp', 'w') as fout:
    for line_i, line in enumerate(lines):
        fout.write(line)
        if line.startswith(CUSTOM_TWO_BODY_START):
            for descriptor in two_body_descriptors:
                parsed_descriptor = _sp.sympify(descriptor.expr)
                fout.write(METHOD_TWO_BODY.format(descriptor.name, 'eval',
                           format_prms(descriptor.num_prms, _sp.ccode(
                                descriptor.expr, user_functions=user_funs))))
                deriv = str(_sp.simplify(
                    _sp.Derivative(parsed_descriptor, rij).doit()))
                deriv = deriv.replace(
                    'Derivative(fcut(rij), rij)', 'dfcut(rij)')
                fout.write(METHOD_TWO_BODY.format(
                    descriptor.name, 'drij',
                    format_prms(descriptor.num_prms,
                                _sp.ccode(deriv, user_functions=user_funs))))

                results = [_sp.simplify(parsed_descriptor),
                           _sp.simplify(_sp.Derivative(
                                parsed_descriptor, rij).doit())]

                simplified_results = [result.replace(
                    'Derivative(fcut(rij), rij)', 'dfcut(rij)') for result
                    in results]
                sub_exprs, simplified_results = _sp.cse(simplified_results)
                method_body = []
                for sub_expr in sub_exprs:
                    method_body.append(
                        'auto {} = {}'.format(
                            sub_expr[0],
                            format_prms(descriptor.num_prms,
                                        _sp.ccode(sub_expr[1],
                                                  user_functions=user_funs))))
                method_body.append('G = {}'.format(format_prms(
                    descriptor.num_prms, _sp.ccode(
                        simplified_results[0], user_functions=user_funs))))
                method_body.append('dGdrij = {}'.format(format_prms(
                    descriptor.num_prms, _sp.ccode(
                        simplified_results[1], user_functions=user_funs))))

                fout.write(EVAL_WITH_DERIVATIVES_TWO_BODY.format(
                    descriptor.name, ';\n  '.join(method_body)))
        elif line.startswith(CUSTOM_THREE_BODY_START):
            for descriptor in three_body_descriptors:
                parsed_descriptor = _sp.sympify(descriptor.expr)
                fout.write(METHOD_THREE_BODY.format(
                    descriptor.name, 'eval',
                    format_prms(descriptor.num_prms,
                                _sp.ccode(descriptor.expr,
                                          user_functions=user_funs))))
                # Derivative with respect to rij
                deriv = _sp.simplify(
                    _sp.Derivative(parsed_descriptor, rij).doit())
                deriv = _sp.sympify(re.sub(
                    'Derivative\(fcut\((?P<arg>.*?)\), (?P=arg)\)',  # NOQA
                    'dfcut(\g<arg>)', str(deriv))).doit()
                fout.write(METHOD_THREE_BODY.format(
                    descriptor.name, 'drij',
                    format_prms(descriptor.num_prms,
                                _sp.ccode(deriv, user_functions=user_funs))))
                # Derivative with respect to rik
                deriv = _sp.simplify(
                    _sp.Derivative(parsed_descriptor, rik).doit())
                deriv = _sp.sympify(re.sub(
                    'Derivative\(fcut\((?P<arg>.*?)\), (?P=arg)\)',  # NOQA
                    'dfcut(\g<arg>)', str(deriv))).doit()
                fout.write(METHOD_THREE_BODY.format(
                    descriptor.name, 'drik',
                    format_prms(descriptor.num_prms,
                                _sp.ccode(deriv, user_functions=user_funs))))
                # Derivative with respect to costheta
                deriv = _sp.simplify(
                    _sp.Derivative(parsed_descriptor, costheta).doit())
                deriv = _sp.sympify(re.sub(
                    'Derivative\(fcut\((?P<arg>.*?)\), (?P=arg)\)',  # NOQA
                    'dfcut(\g<arg>)', str(deriv))).doit()
                fout.write(METHOD_THREE_BODY.format(
                    descriptor.name, 'dcostheta',
                    format_prms(descriptor.num_prms,
                                _sp.ccode(deriv, user_functions=user_funs))))

                # Combined eval and derivatives
                results = [
                    _sp.simplify(parsed_descriptor),
                    _sp.simplify(
                        _sp.Derivative(parsed_descriptor, rij).doit()),
                    _sp.simplify(
                        _sp.Derivative(parsed_descriptor, rik).doit()),
                    _sp.simplify(
                        _sp.Derivative(parsed_descriptor, costheta).doit())]
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
                            sub_expr[0],
                            format_prms(descriptor.num_prms,
                                        _sp.ccode(sub_expr[1],
                                                  user_functions=user_funs))))
                method_body.append('G = {}'.format(format_prms(
                    descriptor.num_prms, _sp.ccode(
                        simplified_results[0], user_functions=user_funs))))
                method_body.append('dGdrij = {}'.format(format_prms(
                    descriptor.num_prms, _sp.ccode(
                        simplified_results[1], user_functions=user_funs))))
                method_body.append('dGdrik = {}'.format(format_prms(
                    descriptor.num_prms, _sp.ccode(
                        simplified_results[2], user_functions=user_funs))))
                method_body.append('dGdcostheta = {}'.format(
                    format_prms(descriptor.num_prms, _sp.ccode(
                        simplified_results[3], user_functions=user_funs))))

                fout.write(EVAL_WITH_DERIVATIVES_THREE_BODY.format(
                    descriptor.name, ';\n  '.join(method_body)))

                # Derivatives with respect to the three arguments
                sub_exprs, simplified_derivs = _sp.cse(results[1::])
                method_body = []
                for sub_expr in sub_exprs:
                    method_body.append(
                        'auto {} = {}'.format(
                            sub_expr[0],
                            format_prms(descriptor.num_prms,
                                        _sp.ccode(sub_expr[1],
                                                  user_functions=user_funs))))
                method_body.append('dGdrij = {}'.format(format_prms(
                    descriptor.num_prms, _sp.ccode(
                        simplified_derivs[0], user_functions=user_funs))))
                method_body.append('dGdrik = {}'.format(format_prms(
                    descriptor.num_prms, _sp.ccode(
                        simplified_derivs[1], user_functions=user_funs))))
                method_body.append('dGdcostheta = {}'.format(
                    format_prms(descriptor.num_prms, _sp.ccode(
                        simplified_derivs[2], user_functions=user_funs))))

                fout.write(DERIVATIVES_THREE_BODY.format(
                    descriptor.name, ';\n  '.join(method_body)))

        elif line.startswith(SWITCH_TWO_BODY_START):
            # Check for hard coded descriptors to offset the ids
            m = re.search('\\(funtype == (\\d)\\)', lines[line_i-1])
            offset = int(m.group(1)) + 1 if m is not None else 0
            for i, descriptor in enumerate(two_body_descriptors):
                if i + offset == 0:
                    fout.write(IF_FUN_ID.format(i + offset, descriptor.name))
                else:
                    fout.write(ELIF_FUN_ID.format(i + offset, descriptor.name))
        elif line.startswith(SWITCH_THREE_BODY_START):
            m = re.search('\\(funtype == (\\d)\\)', lines[line_i-1])
            offset = int(m.group(1)) + 1 if m is not None else 0
            for i, descriptor in enumerate(three_body_descriptors):
                if i + offset == 0:
                    fout.write(IF_FUN_ID.format(i + offset, descriptor.name))
                else:
                    fout.write(ELIF_FUN_ID.format(i + offset, descriptor.name))
        elif line.startswith(GET_TWO_BODY_START):
            m = re.search('== 0\\) id = (\\d);', lines[line_i-1])
            offset = int(m.group(1)) + 1 if m is not None else 0
            for i, descriptor in enumerate(two_body_descriptors):
                if i + offset == 0:
                    fout.write(IF_STRING.format(descriptor.name, i + offset))
                else:
                    fout.write(ELIF_STRING.format(descriptor.name, i + offset))
        elif line.startswith(GET_THREE_BODY_START):
            m = re.search('== 0\\) id = (\\d);', lines[line_i-1])
            offset = int(m.group(1)) + 1 if m is not None else 0
            for i, descriptor in enumerate(three_body_descriptors):
                if i + offset == 0:
                    fout.write(IF_STRING.format(descriptor.name, i + offset))
                else:
                    fout.write(ELIF_STRING.format(descriptor.name, i + offset))

        elif line.startswith(AVAILABLE_TWO_BODY_START):
            # Check for hard coded descriptors to offset the ids
            m = re.search('printf\\("(\\d):', lines[line_i-1])
            offset = int(m.group(1)) + 1 if m is not None else 0
            for i, descriptor in enumerate(two_body_descriptors):
                fout.write('  printf("{}: {}, {}\\n");\n'.format(
                    i + offset, descriptor.name, descriptor.num_prms))
        elif line.startswith(AVAILABLE_THREE_BODY_START):
            m = re.search('printf\\("(\\d):', lines[line_i-1])
            offset = int(m.group(1)) + 1 if m is not None else 0
            for i, descriptor in enumerate(three_body_descriptors):
                fout.write('  printf("{}: {}, {}\\n");\n'.format(
                    i + offset, descriptor.name, descriptor.num_prms))
