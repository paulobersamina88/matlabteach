import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Professor-Style MATLAB + Stiffness Method PRO", layout="wide")

# ---------------------------------
# Utility functions
# ---------------------------------
def solve_system(K, F):
    return np.linalg.solve(K, F)


def assemble_global(n_nodes, connectivity, element_stiffnesses):
    K = np.zeros((n_nodes, n_nodes), dtype=float)
    for e, conn in enumerate(connectivity):
        i, j = conn
        dofs = [i - 1, j - 1]
        K[np.ix_(dofs, dofs)] += element_stiffnesses[e]
    return K


def apply_boundary_conditions(K, F, restrained_dofs_1based):
    restrained = sorted(set([d - 1 for d in restrained_dofs_1based if 1 <= d <= K.shape[0]]))
    free = [i for i in range(K.shape[0]) if i not in restrained]
    Kff = K[np.ix_(free, free)]
    Ff = F[free]
    return Kff, Ff, free, restrained


def expand_displacements(n_dofs, free_dofs, d_free):
    d = np.zeros(n_dofs)
    for i, dof in enumerate(free_dofs):
        d[dof] = d_free[i]
    return d


def reaction_vector(K, d, F):
    return K @ d - F


def matrix_to_latex(M, fmt=".3f"):
    rows = []
    for row in M:
        rows.append(" & ".join([format(float(v), fmt) for v in row]))
    return r"\begin{bmatrix}" + r"\\".join(rows) + r"\end{bmatrix}"


def vector_to_latex(v, fmt=".3f"):
    rows = [format(float(x), fmt) for x in v]
    return r"\begin{Bmatrix}" + r"\\".join(rows) + r"\end{Bmatrix}"


def matlab_matrix(M, name="A"):
    lines = []
    for row in M:
        lines.append(" ".join([f"{float(x):.6g}" for x in row]))
    body = ";\n     ".join(lines)
    return f"{name} = [{body}];"


def matlab_vector(v, name="b"):
    body = "; ".join([f"{float(x):.6g}" for x in v])
    return f"{name} = [{body}]';"


def default_bar_ke(ae_over_l):
    return ae_over_l * np.array([[1.0, -1.0], [-1.0, 1.0]])


def build_demo_system(n_elem, ae_list, l_list, loads):
    n_nodes = n_elem + 1
    connectivity = []
    element_stiffnesses = []
    for e in range(n_elem):
        connectivity.append((e + 1, e + 2))
        element_stiffnesses.append(default_bar_ke(ae_list[e] / l_list[e]))
    K = assemble_global(n_nodes, connectivity, element_stiffnesses)
    F = np.array(loads, dtype=float)
    return n_nodes, connectivity, element_stiffnesses, K, F


# ---------------------------------
# Header
# ---------------------------------
st.title("Professor-Style PRO Teaching App")
st.subheader("MATLAB to Stiffness Matrix Formulation")
st.caption("Interactive teaching tool for structural engineering students: learn the logic, see the matrices, solve the structure, and generate MATLAB code.")

with st.expander("How to use this app"):
    st.markdown(
        """
This app is designed for classroom discussion.

**Recommended teaching flow:**
1. Start in **Learn** to explain the core ideas.
2. Move to **Solve** so students can input values.
3. Open **Visualize** to relate numbers to structural behavior.
4. Show **MATLAB Code** so students see the algorithm.
5. Finish with **Quiz Mode** for recall and discussion.
        """
    )

# ---------------------------------
# Global input panel
# ---------------------------------
st.sidebar.header("Model Setup")
n_elem = st.sidebar.slider("Number of bar elements", 1, 6, 3)
n_nodes = n_elem + 1

st.sidebar.subheader("Element Properties")
ae_list = []
l_list = []
for e in range(n_elem):
    ae = st.sidebar.number_input(f"AE of element {e+1}", value=1000.0, step=100.0, key=f"ae_{e}")
    le = st.sidebar.number_input(f"L of element {e+1}", value=1.0, min_value=0.1, step=0.1, key=f"le_{e}")
    ae_list.append(ae)
    l_list.append(le)

st.sidebar.subheader("Nodal Loads")
loads = []
for i in range(n_nodes):
    fi = st.sidebar.number_input(f"Load at node {i+1}", value=0.0, step=1.0, key=f"f_{i}")
    loads.append(fi)

restrained_text = st.sidebar.text_input("Restrained DOFs (1-based, comma-separated)", value="1")
restrained = [int(x.strip()) for x in restrained_text.split(",") if x.strip().isdigit()]

n_nodes, connectivity, element_stiffnesses, K, F = build_demo_system(n_elem, ae_list, l_list, loads)

solver_ok = True
solver_message = ""
try:
    Kff, Ff, free, restrained0 = apply_boundary_conditions(K, F, restrained)
    d_free = solve_system(Kff, Ff)
    d = expand_displacements(n_nodes, free, d_free)
    R = reaction_vector(K, d, F)
except np.linalg.LinAlgError:
    solver_ok = False
    solver_message = "The reduced stiffness matrix is singular. Check the supports and make sure the structure is stable."
    Kff = Ff = d = R = d_free = None
    free = []

# ---------------------------------
# Main tabs
# ---------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Learn", "Solve", "Visualize", "MATLAB Code", "Quiz Mode"])

# ---------------------------------
# Learn tab
# ---------------------------------
with tab1:
    st.header("Learn the Logic")

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown(
            """
### Big Idea
The stiffness method is simply a **systematic matrix procedure**:

1. Write an element stiffness matrix.
2. Place each element into the correct global location.
3. Apply supports and loads.
4. Solve for unknown displacements.
5. Recover reactions.

This is the same logic used internally by structural software.
            """
        )

        st.latex(r"[K]\{d\}=\{F\}")
        st.markdown(
            """
Where:
- **[K]** = global stiffness matrix  
- **{d}** = nodal displacement vector  
- **{F}** = nodal load vector
            """
        )

    with c2:
        st.info("Teaching cue: tell students that MATLAB is not the lesson by itself. MATLAB is just the language used to automate matrix logic.")

    st.markdown("### Element Stiffness Matrix for an Axial Bar")
    st.latex(r"[k_e] = \frac{AE}{L}\begin{bmatrix}1 & -1 \\ -1 & 1\end{bmatrix}")

    st.markdown("### Current Example")
    elem_df = pd.DataFrame({
        "Element": list(range(1, n_elem + 1)),
        "Start Node": [c[0] for c in connectivity],
        "End Node": [c[1] for c in connectivity],
        "AE": ae_list,
        "L": l_list,
        "AE/L": [ae_list[i] / l_list[i] for i in range(n_elem)]
    })
    st.dataframe(elem_df, use_container_width=True)

    for e, ke in enumerate(element_stiffnesses):
        with st.expander(f"Show element {e+1} stiffness matrix"):
            st.latex(matrix_to_latex(ke))
            st.dataframe(pd.DataFrame(ke), use_container_width=True)

    st.markdown("### Global Stiffness Matrix")
    st.latex(matrix_to_latex(K))
    st.dataframe(pd.DataFrame(K), use_container_width=True)

    if solver_ok:
        st.markdown("### Reduced System After Applying Supports")
        st.latex(r"[K_{ff}]\{d_f\}=\{F_f\}")
        st.latex(matrix_to_latex(Kff))
        st.latex(vector_to_latex(Ff))
    else:
        st.error(solver_message)

# ---------------------------------
# Solve tab
# ---------------------------------
with tab2:
    st.header("Solve the Structure")
    st.markdown("Change the sidebar inputs and observe how the structural response changes.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Input Vectors and Matrices")
        st.write("Global stiffness matrix")
        st.dataframe(pd.DataFrame(K), use_container_width=True)
        st.write("Load vector")
        st.dataframe(pd.DataFrame(F, columns=["F"]), use_container_width=True)

    with c2:
        st.subheader("Support Information")
        st.write(f"Restrained DOFs: {restrained if restrained else 'None'}")
        st.write(f"Free DOFs: {[i + 1 for i in free] if solver_ok else 'Not available'}")

        if solver_ok:
            st.success("Stable system solved successfully.")
        else:
            st.error(solver_message)

    if solver_ok:
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Displacements")
            st.dataframe(pd.DataFrame({"DOF": np.arange(1, n_nodes + 1), "Displacement": d}), use_container_width=True)
        with c4:
            st.subheader("Reactions")
            st.dataframe(pd.DataFrame({"DOF": np.arange(1, n_nodes + 1), "Reaction": R}), use_container_width=True)

        with st.expander("Step-by-step professor explanation"):
            st.markdown(
                f"""
**Step 1:** Build each element stiffness matrix using \\(AE/L\\).  
**Step 2:** Assemble them into the full \\([K]\\) using connectivity.  
**Step 3:** Remove restrained DOFs {restrained if restrained else '[]'} to get the reduced system.  
**Step 4:** Solve the reduced displacement vector.  
**Step 5:** Expand back to the full displacement vector and compute reactions using \\([K]\\{{d\\}}-\\{{F\\}}\\).
                """
            )

# ---------------------------------
# Visualize tab
# ---------------------------------
with tab3:
    st.header("Visualize Structural Response")

    if solver_ok:
        c1, c2 = st.columns(2)

        with c1:
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot(range(1, n_nodes + 1), d, marker="o")
            ax1.set_xlabel("Node")
            ax1.set_ylabel("Displacement")
            ax1.set_title("Nodal Displacement Shape")
            ax1.grid(True)
            st.pyplot(fig1)

        with c2:
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.bar(range(1, n_nodes + 1), R)
            ax2.set_xlabel("Node")
            ax2.set_ylabel("Reaction")
            ax2.set_title("Reaction Distribution")
            ax2.grid(True)
            st.pyplot(fig2)

        st.subheader("Conceptual Deformed Shape")
        scale = st.slider("Display scale factor", min_value=1.0, max_value=100.0, value=20.0, step=1.0)
        x = np.arange(1, n_nodes + 1)
        y0 = np.zeros(n_nodes)
        yd = d * scale

        fig3, ax3 = plt.subplots(figsize=(10, 3))
        ax3.plot(x, y0, marker="o", label="Original")
        ax3.plot(x, yd, marker="o", label="Deformed")
        for i in range(n_nodes - 1):
            ax3.plot([x[i], x[i+1]], [y0[i], y0[i+1]])
            ax3.plot([x[i], x[i+1]], [yd[i], yd[i+1]])
        ax3.set_xlabel("Node Position")
        ax3.set_ylabel("Relative Displacement")
        ax3.set_title("Original vs Deformed Configuration")
        ax3.grid(True)
        ax3.legend()
        st.pyplot(fig3)

        st.info("Teaching cue: ask students why the shape changes when AE/L changes. This helps them connect stiffness to physical behavior.")
    else:
        st.error(solver_message)

# ---------------------------------
# MATLAB Code tab
# ---------------------------------
with tab4:
    st.header("Generated MATLAB Code")
    st.markdown("Use this as a bridge from classroom interaction to coding practice.")

    matlab_lines = []
    matlab_lines.append("% Professor-Style MATLAB script generated from the Streamlit app")
    matlab_lines.append(f"nElem = {n_elem};")
    matlab_lines.append(f"nNodes = {n_nodes};")
    matlab_lines.append("")

    for e in range(n_elem):
        matlab_lines.append(f"AE({e+1}) = {ae_list[e]:.6g};")
        matlab_lines.append(f"L({e+1}) = {l_list[e]:.6g};")
    matlab_lines.append("")

    matlab_lines.append("K = zeros(nNodes,nNodes);")
    matlab_lines.append("for e = 1:nElem")
    matlab_lines.append("    ke = (AE(e)/L(e)) * [1 -1; -1 1];")
    matlab_lines.append("    nodes = [e e+1];")
    matlab_lines.append("    K(nodes,nodes) = K(nodes,nodes) + ke;")
    matlab_lines.append("end")
    matlab_lines.append("")
    matlab_lines.append(matlab_vector(F, "F"))
    matlab_lines.append(f"restrained = [{ ' '.join(str(r) for r in restrained) }];")
    matlab_lines.append("allDofs = 1:nNodes;")
    matlab_lines.append("free = setdiff(allDofs, restrained);")
    matlab_lines.append("Kff = K(free,free);")
    matlab_lines.append("Ff = F(free);")
    matlab_lines.append("df = Kff\\Ff;")
    matlab_lines.append("d = zeros(nNodes,1);")
    matlab_lines.append("d(free) = df;")
    matlab_lines.append("R = K*d - F;")

    matlab_script = "\n".join(matlab_lines)

    st.code(matlab_script, language="matlab")

    with st.expander("Show current matrices in MATLAB-ready form"):
        st.code(matlab_matrix(K, "K"), language="matlab")
        st.code(matlab_vector(F, "F"), language="matlab")
        if solver_ok:
            st.code(matlab_matrix(Kff, "Kff"), language="matlab")
            st.code(matlab_vector(Ff, "Ff"), language="matlab")

# ---------------------------------
# Quiz Mode tab
# ---------------------------------
with tab5:
    st.header("Quiz Mode")
    st.markdown("Use these for recitation, discussion, or quick classroom checks.")

    q1 = st.radio(
        "1. In the stiffness equation [K]{d} = {F}, what does {d} represent?",
        ["Applied loads", "Nodal displacements", "Element length", "Reaction forces"],
        key="q1"
    )
    if st.button("Check Q1"):
        if q1 == "Nodal displacements":
            st.success("Correct.")
        else:
            st.error("Not correct. {d} is the displacement vector.")

    q2 = st.radio(
        "2. What is the main reason for applying boundary conditions?",
        ["To increase the number of DOFs", "To represent supports and restraints", "To remove all loads", "To avoid matrices"],
        key="q2"
    )
    if st.button("Check Q2"):
        if q2 == "To represent supports and restraints":
            st.success("Correct.")
        else:
            st.error("Not correct. Boundary conditions model the supports and restraints.")

    q3 = st.radio(
        "3. Why do we assemble element stiffness matrices into a global matrix?",
        ["Because one element is enough for the whole structure", "To combine the behavior of all connected elements", "To avoid using MATLAB", "To remove reactions"],
        key="q3"
    )
    if st.button("Check Q3"):
        if q3 == "To combine the behavior of all connected elements":
            st.success("Correct.")
        else:
            st.error("Not correct. Assembly builds the whole structural system from all element contributions.")

    with st.expander("Professor discussion prompts"):
        st.markdown(
            """
- What happens if you increase AE of one element only?
- What happens if the middle node carries the load instead of the last node?
- Why does a missing support create a singular matrix?
- Which part of this procedure is done automatically by STAAD or ETABS?
- If this were a beam or frame, what would become more complicated?
            """
        )

st.markdown("---")
st.caption("PRO classroom version: interactive, visual, code-linked, and discussion-ready.")
