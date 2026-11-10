# Andy Lomas — "Cellular Forms: an Artistic Exploration of Morphogenesis"

**Complete extraction / reimplementation reference.**

| Field | Value |
|---|---|
| Title | Cellular Forms: an Artistic Exploration of Morphogenesis |
| Author | Andy Lomas (footnoted ¹) |
| Venue | AISB50 (per task metadata; the page renders carry no visible running header/footer with venue or DOI) |
| Length | 8 pages, two-column layout |
| Source images | `docs/reference/pages/page_01.png` … `page_08.png` (1666×2358) and `page_03_hi.png`, `page_04_hi.png` (2975×4210) |
| Extraction method | Every page read as an image; all formulas, the Figure 4 diagram, all four Figure 7 panels, Table 1, the footnote and the reference list additionally verified at 2×–4× zoom crops of the hi-res renders |

Page map: p1 = title, author, footnote, Abstract, Figure 1, intro paragraph, §1 Introduction, §2 Related Work (start). p2 = §2 (end), Figure 2, previous-work paragraphs, §3 Cellular Forms, §4 Mathematical Model (start), Figure 3. p3 = §4 model text, equations (1)–(3), Figure 4, bulge equations, position update, Figure 5. p4 = Figure 6, repulsion + collisionOffset equation, food-level growth methods, Figure 7, cell-division text, simplification list, §5 Implementation (first sentence). p5 = §5 Implementation, Table 1, rendering text, §6 Artistic Artefacts, Figure 8, rendering style 1. p6 = Figure 9, rendering style 2, §7 Results, Figure 10, results text. p7 = results end, Figure 11, §8 Conclusions & Future Work, Figure 12, future-work paragraphs, REFERENCES heading, reference [1]. p8 = references [2]–[20] (single narrow column; the only content on the page).

---

## Abstract (verbatim)

> **Abstract.** Cellular Forms uses a simplified model of cellular growth to generate intricate sculptural shapes. Structures are created out of interconnected cells, with rules for the forces between cells, as well as rules for how cells accumulate internal nutrients. When the nutrient level in a cell exceeds a given threshold the cell splits into two, with both the parent and daughter cells reconnecting to their immediate neighbours. Many different complex organic structures are seen to arise from subtle variations of these rules, creating forms with strong reminiscences of plants, corals, internal organs and micro-organisms.

(Verified at zoom: "strong **reminiscences** of plants", not "resemblances".)

Unnumbered paragraph after Figure 1 (verbatim):

> The aim is to create structures emergently: exploring generic similarities between many different forms in nature rather than recreating any particular organism, and in the process exploring universal archetypal forms that can come from growth processes rather than top-down externally engineered design.

---

## 1 Introduction (verbatim)

> The use of simulation methods in generative art can be seen as a natural extension of systems art [1], where the artist defines a process that can be run autonomously to create artefacts. With a sufficiently rich simulation system there is an expectation that surprising emergent results can be generated which would be difficult, or potentially even impossible, to create without the aid of digital technology. The process can be seen as one of exploration: both in defining the rules for the simulation systems and exploring the range of results that can be achieved once a system has been created.
>
> In particular, simulation systems that are inspired by biological processes, such as morphogenesis, can be used as a powerful means to explore the nature of organic form. Can the astonishingly complex forms that are seen in nature emerge from simple rules? This can be viewed as exploring the nature of the basic fabric available to create structures when they are generated as a result of growth processes.
>
> This paper describes Cellular Forms, an exploration of how rich evocative forms can be created using a simplified biological model of cellular growth. The model used is a deliberately simplified one, both to explore how a simple model can create richly emergent results, and to be computationally sufficiently efficient to allow the creation of structures with many millions of cells in order to achieve a high level of complexity and detail with the aim of evoking a powerful aesthetic result.

---

## 2 Related Work (verbatim)

> The relationship between growth and form has been the subject of study for many years. Major influences behind the work described here are Ernst Haeckel's studies of forms in nature [2] and D'Arcy Thompson's seminal "On Growth and Form" [3].
>
> Alan Turing's paper "The Chemical Basis of Morphogenesis" [4] can be seen as the origin of using digital simulation methods to examine potential mechanisms behind pattern generation and growth, performing biological experiments "in silico" rather than "in vitro" or "in vivo". The reaction-diffusion equations that Turing describes in his paper can create a surprisingly rich range of complex patterns, with remarkable resemblance to many of those seen in nature such as pigmentation patterns on the coats of leopards, zebra and angelfish. Variations on reaction-diffusion equations, particularly following the work of Greg Turk [5], are commonly used in computer graphics to create convincingly biological textures for creatures.
>
> Probably the most common method utilised to create convincing biological structures is the use of L-Systems, as originally proposed by Aristid Lindenmayer [6]. Rules for branching and the relative sizes of segments between branches are expressed using a simple grammar with a generation rule that can be recursively applied to the form to create the next level. L-Systems are commonly used in computer graphics to produce tree-like forms, and can be a very efficient way to create complex structures, but effects like branching are explicitly encoded into the system rather than arising emergently from lower level processes.

(text continues across the page 1 → page 2 column break at "explicitly | encoded")

> Previous work by the author explored structures that can be created by variations on diffusion-limited aggregation [7], used as a simplified model of a growth system [8], [9]. This represents growth by repeated deposition: the structure is initialised with a single seed particle. Successive new particles are allowed to randomly move in an external medium until they collide with the structure generated so far. They then become attached to the structure at that position, and the process is repeated. The results of this Aggregation series were a range of structures with reminiscences of dendritic plants and finely branched corals.
>
> The previous works most related to the work described here are Jaap Kaandorp's work on "Accretive Growth" [10] [11] and George Hart's "Growth Forms" [12]. Both of these use a model where cells are described by a surface of linked particles, with rules for when these cells split and how the topology of the surface changes after cell division. In Kaandorp's work the aim is to mimic the growth of coral-like forms, with growth in different areas based on external nutrient gradients that mimic the availability of water-borne food for marine organisms. In Hart's work some particles are designated as 'buds' which are used to explicitly control and stimulate growth in local areas, causing effects such as branching. The results of Kaandorp and Hart's work are a variety of branched coral-like structures, and the simulations described are run for a few thousand cell primitives.

---

## 3 Cellular Forms (verbatim)

> In Cellular Forms the principal aim was to create a system capable of generating complex biologically evocative forms based on the simulation of growth by cellular division.
>
> Following the author's previous work, it was desired that the model should be flexible enough that it should be capable of producing results similar to those from the 'Aggregation' series as well as creating additional structures not achievable by that simulation framework. The aim was to be exploratory rather than create any specific target forms, but to be capable of producing structures reminiscent of internal organs such as the folded surface of brains.
>
> In order to create this greater range of possible forms it was decided to use a model based on a simplified version of morphogenesis through cellular division. The system should be capable of creating complex sheets of cells, with rules governing when cells divide, how the topology of the surface of cells is affected by the newly created cells, and creating forces between cells to induce the surface to fold into complex shapes.
>
> By having a variety of different methods to induce growth it was hoped to be able to create the extended range of structures desired. If growth was stimulated by external randomly transported food particles which directly cause the first cell they hit to divide, it should be capable of creating similar effects to those seen in the Aggregation series. On the other hand, if growth was based on concentrations of chemicals diffusing through the structure, potentially with all cells receiving the same amount of nutrient, then it was hoped that structures more like internal body organs could be produced.
>
> It was also important that the simulation system should be capable of generating many millions of cell primitives. This was desired in order to achieve a compelling level of intricate detail in the structures, and had to be achievable within the limitations of available conventional PC hardware.

(column break mid-sentence: "…it should be capable of | producing results similar to…")

---

## 4 Mathematical Model (verbatim, with equations)

> The model used is based on a simplified version of cells. A particle system [13] representation is used, with each cell represented by one particle, and each particle linked to a number of other particles that it is directly attached to.
>
> While the structures are three-dimensional, the topology of the connected particles is that of a two-dimensional surface. In all the structures illustrated here, the system starts with a simple ball of cells with all the cells uniformly spaced on the surface of a sphere.
>
> Development of the form proceeds by a combination of forces that mediate interactions between the cells, and cell divisions that change the topology of the structure. The simulation takes place over time, with time incrementing in uniform clock cycle steps.
>
> Cells that are directly linked try to maintain a constant distance from each other. Additional rules try to restore the sheet to a locally planar state if there is a fold in the surface, and to bulge the sheet out when links are in compression. The intention is that these two influences will work in competition with each other, with different strength factors for each tending to create surfaces with a variety of characteristics.
>
> The actual implementation of all these effects is achieved by calculating a new target position for each of these influences, and offsetting the cell's position towards the new target position after multiplying by a restoring factor. Values for these factors are parameters for the simulation system as a whole.
>
> Consider a system with simulation parameters `linkRestLength`, `springFactor`, `planarFactor` and `bulgeFactor`. Let a cell have position **P** which is linked to *n* particles with positions **L**ᵣ, and the unit length normal to the surface at the current cell position is **N**. The target positions for three different influencing effects are calculated by the following methods:
>
> 1. The tendency for linked cells to maintain a constant distance from each other is implemented using a linear spring-like system. The target position for the springs is calculated by taking the average of the rest positions that each link would push the particle to if it were the only influence:
>
> **springTarget** = (1/n) Σ_{r=1..n} ( **L**ᵣ + linkRestLength × hat(**P** − **L**ᵣ) )
>
> 2. The planar target position, that acts in a similar manner to a torsion spring, is simply calculated by taking the average of all the positions of directly linked particles. This is designed to have the effect of tending to reduce folds and bumps in the surface, restoring the surface to a local planar state:
>
> **planarTarget** = (1/n) Σ_{r=1..n} **L**ᵣ
>
> [Figure 4 diagram here]
>
> **Figure 4.** Diagram illustrating calculation of the bulge target position.
>
> 3. The bulge target position is determined by calculating the distance that each link would have to push the particle outwards along the direction of the surface normal in order to restore the link to its rest length. This is designed to have an effect of tending to bulge the surface outwards in the direction of the normal when links are in compression. The bulge distance due to each link is calculated in a simple application of the cosine formula for triangles, and the average taken for all the links to create the desired overall distance *bulgeDist* in the direction of the normal. The **bulgeTarget** vector is then taken by going *bulgeDist* in the direction of the surface normal from **P**.
>
> dotNᵣ = ( **L**ᵣ − **P** ) · **N**
>
> *bulgeDist* = (1/n) Σ_{r=1..n} ( √( linkRestLength² − |**L**ᵣ|² + dotNᵣ² ) + dotNᵣ )
>
> **bulgeTarget** = **P** + *bulgeDist* × **N**
>
> The new position for the particle position **P**′ is then calculated by offsetting **P** in the direction of each of these target position using the three different simulation factor values.
>
> **P**′ = **P** + springFactor × ( **springTarget** − **P** ) + planarFactor × ( **planarTarget** − **P** ) + bulgeFactor × ( **bulgeTarget** − **P** )
>
> One thing to notice about this is that there is no momentum term: the new position is simply calculated by moving it a fraction of the distance towards the target positions controlled by the restoring factors. This can be justified if we consider the cells to be growing in a medium that is highly viscous relative to the cell size, so there is a large damping effect on any velocities. It also has the advantage of making the system less prone to unstable oscillations, particularly when we are in effect adding energy to the system every time cells divide.

(column break: §4 item 3 begins the right column of page 3; "target position" — trailing "s" not resolvable at available resolution, see OCR-risk log.)

The repulsion model (page 4, verbatim; note Figure 6 and its caption sit at the top of the left column, before this text):

> Cells that aren't directly linked to each other but are in close proximity experience a repulsive influence. Without this it was found that structures would degenerate into an incoherent state as they started to generate folds. In effect the repulsive influence between cells imposes a constraint of structural coherence on the form. An analogy can be drawn with D'Arcy Thompson's arguments for how physics imposes constraints on the possible shapes for forms created by growth processes.
>
> This repulsive influence is controlled by two other parameters for the simulation system which define a radius of influence (*roi*) and *repulsionStrength*. The effect of the repulsion is applied by calculating a **collisionOffset** vector to be added to each particle's position where
>
> **collisionOffset** = repulsionStrength Σ_{r∈A} ( (roi² − |**P** − **P**ᵣ|²)/roi² × hat(**P** − **P**ᵣ) )
>
> and *A* is the set of all particles within the radius of influence of the current particle that aren't directly linked to the current particle. Directly linked particles are excluded from the repulsion calculations since they are considered directly attached to each other, and the influences between them are already controlled by the previously described other effects.
>
> Each cell has an internal 'food level.' The food accumulates in a cell and when it exceeds a given threshold the cell is selected for splitting. Various different methods have been implemented to affect food levels, and therefore affect growth rates, including:
>
> - Uniformly adding a random amount to each cell at each time step.
> - Using reaction-diffusion equations (RDEs) over the surface [5] to create differential areas of growth. The RDE is calculated using the cells as places where chemicals are stored, and diffusing the chemicals along the direct links between cells. One of the RDE chemicals is used as a 'nutrient level,' affecting how much the food level in the cells increments at each time step.
> - Using ray-tracing to simulate light coming in from outside the structure which stimulates nutrient creation in the cells that the light hits. This nutrient is used to control the food increment in each cell, and can also diffuse from one cell to another along the links between cells to distribute the nutrient through the structure.

Cell division (page 4 right column, verbatim):

> When a cell has been selected for splitting, a number of parameters are used to control how the split occurs. These are used to control how the topology of links between the cells is changed by cell division. First, two links are chosen that represent the plane of cleavage. All the links to one side of the plane of cleavage are left connected to the parent cell, while the links to the other side are disconnected from the parent and replaced with links to the daughter cell. Along the plane of cleavage links are made to both the parent and daughter cells. A new link is also created directly between the parent and daughter.

Key simplifications (page 4 right column, verbatim list):

> There are some key simplifications in this system compared with one that aims to accurately mimic biological processes. In particular:
>
> - Apart from the potential for differential growth rates as described above, there is no cell differntiation [*sic*]. The parameters governing forces between cells and how cell division takes place are constants for the whole system, and don't change over time. The target link length is also the same uniform constant value for all links between any cells.
> - Cells are represented by simple spheres with only a position and radius. They don't have any principal axis or orientation.
> - There is no cell motility. The links that describe attachments between cells are formed immediately following cell division.
> - As previously mentioned, the cells are always connected together in arrangements that are topologically equivalent to a simple closed surface rather than being a solid volume.

("differntiation" is printed exactly so — missing an "e"; verified at 3× zoom.)

---

## 5 Implementation (verbatim)

> Typical simulations are run for tens of thousands of iterations with data sets growing to over 50 million particles.
>
> The software to run the simulations and render images from the data sets created is implemented in C++ and CUDA [14].
>
> To make use of the general purpose parallel processing capabilities of modern graphics hardware, all the calculations involving the individual particles are executed using CUDA on the GPU. This includes all the functions that simulate the effects of forces between the particles, cast rays into the structure to render images or simulate light rays, and handle topological changes to the surface that occur when cells divide. This means that all the data for the cellular structures can be kept purely in GPU memory, avoiding the need to transfer large amounts of data across the PCI bus between the host (CPU) memory and the device (GPU) memory. In the current implementation the only need for transferring cell data between the CPU and GPU is if it is required to write out data to a file on disk, or to read data back from disk.
>
> In particular, implementing using the GPU allowed a very significant speed improvement for the calculations of repulsive interactions between cells in close proximity but not directly linked to each other. The previous implementation of this on the CPU was too slow to make it feasible to run simulations with the required number of cells.

**Table 1** (transcribed in full below, in its own section). The paragraph that follows (verbatim):

> The software is implemented as an extensible framework. An object oriented approach was used with a hierarchy of inheriting classes representing the data for the structures. This starts with a base class for representing general particle data on the GPU, which is then specialised through a series of derived classes that inherit from each other up until the final class used in all the simulations shown here.
>
> Rendering 2D images from the cellular simulation data is done using ray-tracing techniques, with the cells treated as spheres with radii based on the average distances to linked cells. A number of different techniques are implemented, including producing solid surface renders with 'ambient occlusion' for the shading which represents the effects of a self-shadowing from a uniform omnidirectional diffuse light, and "X-Ray" renders that treat each sphere as a contributor to an accumulated density calculated by tracing a ray though [*sic*] the whole structure. All the key rendering functions are implemented using CUDA kernels on the GPU, both for speed and so that all the particle data can be kept solely on the GPU.

("though" is printed exactly so — i.e. "through" with a dropped "r"; verified at 4× zoom. Note "omnidirectional" unhyphenated here, but "omni-directional" hyphenated in §6.)

---

## 6 Artistic Artefacts (verbatim)

> Simulations are first run with a maximum of a million cells to create quick initial sample tests of simulation parameter values which may create interesting results. These tests typically take between 1 and 5 minutes to run for each sample. From these, candidates are selected to run full length simulations and high-resolution renders. These final simulations are executed to create the most detailed structures possible, which typically means between 52 million and 56 million cells before the memory limits of the current hardware (NVIDIA GTX Titan with 6GB RAM) are reached. These full length simulations typically take between 1 and 4 hours each.
>
> Rendered images are created directly from the simulations, which are used to make the artistic artefacts from the series. These artefacts can take various forms:
>
> - High resolution prints of the final structures. Currently these are rendered at 8600 by 8600 pixels, to allow very fine detail when printing at large sizes.
> - High definition animations taken by rendering images at equally spaced time intervals, to show how the structures develop incrementally over time.
> - 3D stereo views taken by rendering pairs of images using cameras with an interocular separation.
>
> Currently, two main rendering styles are used for the final artefacts:
>
> [Figure 8]
>
> 1. Solid surfaces that use 'ambient occlusion' self-shadowing to reveal the final forms. Uniform omni-directional diffuse light is used on a surface with Lambertian (ideal diffuse) reflectance to create a directionally unbiased view of the surface. The images produced in this manner emphasise sculptural forms of the surfaces, and echo the illustrative techniques that Ernst Haeckel used in his studies of nature [2].

(page 6, continuation):

> 2. 'X-Ray' images that show accumulated density through the whole structures. These images reveal complex internal structures not apparent in the external solid renders, and have strong resemblances to biological studies using microscopy.

---

## 7 Results (verbatim)

> The structures produced by the simulations are all simple closed surfaces, topologically equivalent to the surface of a sphere. However, they become incredibly intricately folded due to tensions in the surface and forces between the cells as the structure develops over time.
>
> The original intention behind creating this system was to be able to create a greater range of emergent forms than had been seen with previous work that explored variations of diffusion-aggregation as a simplified model of growth systems [8], [9].
>
> The results have exceeded the initial expectations. Many rich behaviours appear to emerge such as complex folding, branching, rhythmic pulses and waves of growth propagating across surfaces. These can be particularly observed in the animations created from the simulations [15], [16]. Many of these effects appear to have deep resemblances to behaviour seen in nature such as pulses of growth during embryonic cell division [17] and mitotic division waves [18].
>
> One thing to note is that all the structures seen here appear to emerge without the need for differentiation into different cell types. The system framework allows reaction-diffusion equations to be used to create regions of cells with different growth rates, but it was found that these influences could be completely omitted and still produce rich results. All the forms shown here were created without the use of RDEs, only using either uniform growth rates for all cells, or having nutrient for cell growth created by incident light rays which could diffuse throughout the structure.

(page 6 right column, continuing to page 7):

> In general, using uniform cell growth rates appears to produce results that look more like internal organs, such as brains, with complex folded shapes. Using incident light rays to create nutrient produces a larger range of structures. This includes the internal organ-like structures when the inter-cellular diffusion rate for nutrient diffusion rates are high, but when it is set low and the light comes from a single direction we appear to naturally get remarkably plant-like forms.
>
> One of the most necessary contributions to create interesting structurally coherent forms appears to be the use of repulsion influences between cells which are in close proximity but not directly linked to each other. Without these the forms tend to degenerate into a meaningless unstructured mess, as there is nothing preventing the surface self-intersecting as it folds. These additional influences can be seen as exerting physical constraints on the structures.

(The awkward phrase "when the inter-cellular diffusion rate for nutrient diffusion rates are high" is exactly what is printed — verified at 4× zoom.)

---

## 8 Conclusions & Future Work (verbatim)

> The use of a simple system of interactions between adjacent or spatially close cells is seen to produce a wide range of complex results.
>
> [Figure 12]
>
> Many of the surface patterns produced, such as protuberances and brain-like folds, are reminiscent of the sort of shapes produced by reaction-diffusion equations. It is potentially interesting that these are seen to occur without the need for any explicit influencers reacting together and diffusing between the cells.
>
> The framework used here is an extensible one and there are a number of interesting future directions that could be taken.
>
> The current model doesn't have any active cell differentiation, with the simulation parameters set globally for the whole system. Cell differentiation could be explored in a number of different ways, such as by having different simulation parameters for each cell. One simple idea would be to have two sets of parameters, and a gradient between the cells which is used to blend between those parameter values for each cell. It may also be interesting to re-introduce reaction-diffusion equations to control the blending between the parameters instead of having a single simple gradient.
>
> Additional or alternative implementations of the forces between cells could be explored. It may be reasonable to expect that many of the structures seen here are generic enough to not be the results of specific details of the implementations used, but it would be interesting to see if the types of structures generated do change if modifications are made to the inter-cellular forces.
>
> Currently the main limit to the complexity of the structures created is the available memory in the GPU to hold the cell data. With increasing hardware capabilities it will be possible to run larger simulations. The algorithms used could also be changed to remove the requirement that all particles are held concurrently in GPU memory, though this may have impractical performance implications, or be altered to distribute calculations between multiple GPUs.
>
> The effect of other influences and forces on the system could be explored. These could be readily integrated into the system using similar mechanisms to the current implementation of repulsion between cells. Interesting sources could be fluid flows, surfaces that exert their own attraction or repulsion forces, or the cells themselves acting as primitives that can generate their own complex electrical or magnetic force fields which affect subsequent growth.
>
> It would be interesting to explore the effects of cell death as well as cell growth, with appropriate topological changes to the surface when cells die. These could include rules that allow holes to be created in the surface, resulting in the surface no longer being equivalent to the surface of a sphere.
>
> The software framework could also be extended to model cells in volumetric arrangements, such as by using tetrahedral meshes. However, since the prime goal here was to create interesting sculptural shapes, the use of a surface model was probably appropriate. This can also be justified by taking into consideration how epithelial cell tissues in real biological systems undergo morphogenesis.
>
> Other rendering techniques could be used. One alternative that was explored in early tests was to use a variation on Voronoi cells [19], to illustrate the cell walls [20]. This isn't currently implemented on the GPU, but it should be a relatively easy addition to the code used for rendering.
>
> It would be fascinating to see if 3D printing could be used to turn these structures into physical sculptural form, revealing their full three-dimensional complexity. Some of the more detailed dendritic forms are likely to prove challenging, but it should be an interesting challenge.

("impractical performance implications" verified at 4× zoom; "equivalent" verified at 4× zoom.)

---

## Formulas

Typography as printed: vectors are bold upright (**P**, **L**ᵣ, **N**, **P**ᵣ, **springTarget**, **planarTarget**, **bulgeTarget**, **P**′, **collisionOffset**); scalars are italic (*n*, *r*, *bulgeDist*, *roi*, and the parameter names *linkRestLength*, *springFactor*, *planarFactor*, *bulgeFactor*, *repulsionStrength*, which are printed in italic in the equations but roman/mono in running text). All six equations are displayed, numbered only by their position in the enumerated list (1)–(3) for the three targets; the dotN/bulgeDist/bulgeTarget triple and **P**′ and **collisionOffset** are unnumbered display equations.

### 1. springTarget (spring-like link restoration)

$$\mathbf{springTarget} = \frac{1}{n}\sum_{r=1}^{n}\left(\mathbf{L}_r + \mathrm{linkRestLength} \times \widehat{\left(\mathbf{P} - \mathbf{L}_r\right)}\right)$$

Notation: sum over all *n* links, r = 1…n; the wide hat denotes the unit vector in the direction of **P** − **L**ᵣ, i.e. pointing from the linked particle **L**ᵣ towards the current particle **P**; `linkRestLength` is the scalar rest length. Each term is the rest position the particle would be pushed to by link *r* alone (lying on the ray from **L**ᵣ through **P** at distance `linkRestLength` from **L**ᵣ); the average over links is the spring target. Verified verbatim at 2× zoom from the hi-res page 3.

### 2. planarTarget (torsion-spring-like planarity)

$$\mathbf{planarTarget} = \frac{1}{n}\sum_{r=1}^{n}\mathbf{L}_r$$

Notation: plain average (centroid) of the positions of the *n* directly linked particles; no hats, no weights. Acts "in a similar manner to a torsion spring", reducing folds and bumps to restore a locally planar state.

### 3. dotNᵣ (normal component of link r)

$$\mathrm{dotN}_r = \left(\mathbf{L}_r - \mathbf{P}\right)\cdot\mathbf{N}$$

Notation: scalar (dot) product of the link vector (from **P** to the linked particle **L**ᵣ) with the unit surface normal **N**; i.e. the signed length of the link's projection onto the normal direction (positive when **L**ᵣ is on the outward side of **P**).

### 4. bulgeDist (average outward push distance) — CRITICAL, see verification below

$$\mathrm{bulgeDist} = \frac{1}{n}\sum_{r=1}^{n}\left(\sqrt{\mathrm{linkRestLength}^2 - \left|\mathbf{L}_r\right|^2 + \mathrm{dotN}_r^2} + \mathrm{dotN}_r\right)$$

Notation: sum over r = 1…n; the square root covers exactly `linkRestLength² − |L_r|² + dotN_r²` (verified at 3× zoom: the radical bar ends before the closing parenthesis); the `+ dotN_r` term is outside the root but inside the summed parentheses; the whole sum is multiplied by 1/n. **As printed, the second term under the root is |L_r|² (squared norm of the linked particle's absolute position). The geometrically correct quantity is |L_r − P|² (squared current link length) — full analysis in "Formula verification" below.**

### 5. bulgeTarget

$$\mathbf{bulgeTarget} = \mathbf{P} + \mathrm{bulgeDist}\times\mathbf{N}$$

Notation: the current position **P** displaced by the scalar distance *bulgeDist* along the unit surface normal **N** (outward).

### 6. P′ (position update)

$$\mathbf{P}' = \mathbf{P} + \mathrm{springFactor}\times\left(\mathbf{springTarget}-\mathbf{P}\right) + \mathrm{planarFactor}\times\left(\mathbf{planarTarget}-\mathbf{P}\right) + \mathrm{bulgeFactor}\times\left(\mathbf{bulgeTarget}-\mathbf{P}\right)$$

Notation: explicit Euler-style move of **P** a fraction of the way towards each of the three targets, with the three scalar restoring factors `springFactor`, `planarFactor`, `bulgeFactor`. **No momentum/velocity term** — the paper explicitly notes this (justified by a highly viscous medium; also makes the system less prone to unstable oscillations given that division adds energy).

### 7. collisionOffset (repulsion between non-linked close cells)

$$\mathbf{collisionOffset} = \mathrm{repulsionStrength}\sum_{r\in A}\left(\frac{roi^2 - \left|\mathbf{P}-\mathbf{P}_r\right|^2}{roi^2}\times\widehat{\left(\mathbf{P}-\mathbf{P}_r\right)}\right)$$

Notation: sum over the set *A* = all particles **P**ᵣ within the radius of influence *roi* of the current particle that are **not** directly linked to it (linked particles are excluded — their interactions are handled by the spring/planar/bulge effects); the wide hat denotes the unit vector in the direction **P** − **P**ᵣ (pointing away from the neighbour, i.e. repulsive); the scalar weight (roi² − |P − P_r|²)/roi² = 1 − |P − P_r|²/roi² equals 1 at zero separation and falls linearly (in squared distance) to 0 at the radius of influence; the whole sum is scaled by `repulsionStrength`. The result is **added** to the particle's position. Verified verbatim at 2× zoom from the hi-res page 4 (hat over (**P** − **P**ᵣ), subscript r ∈ A under the Σ confirmed).

---

## Formula verification: the bulgeDist ambiguity

**Printed formula (page 3, verbatim):**

$$\mathrm{bulgeDist} = \frac{1}{n}\sum_{r=1}^{n}\left(\sqrt{\mathrm{linkRestLength}^2 - \left|\mathbf{L}_r\right|^2 + \mathrm{dotN}_r^2} + \mathrm{dotN}_r\right)$$

**Question:** is the `|L_r|²` under the root really the squared norm of the linked particle's absolute position, or must it geometrically be `|L_r − P|²`, the squared current link length?

**Conclusion: it must be `|L_r − P|²`. The printed `|L_r|²` is a typographical error (a dropped "− P"), and any reimplementation must use the squared current link length.** Reasoning:

1. **Derivation from the stated intent.** The text defines the per-link bulge distance as "the distance that each link would have to push the particle outwards along the direction of the surface normal in order to restore the link to its rest length". Let x be that distance, T = P + x·N the pushed position. Restoring link r to rest length means |L_r − T| = linkRestLength. Expanding with dotN_r = (L_r − P)·N and d_r = |L_r − P|:

   |L_r − P − x·N|² = d_r² − 2x·dotN_r + x² = linkRestLength²
   ⟹ x² − 2·dotN_r·x + (d_r² − linkRestLength²) = 0
   ⟹ x = dotN_r ± √(dotN_r² − d_r² + linkRestLength²)

   The outward root — the one that "pushes the particle outwards" — is x_r = dotN_r + √(linkRestLength² − |L_r − P|² + dotN_r²), which is *exactly* the printed term with `|L_r|²` replaced by `|L_r − P|²`. Averaging the x_r over links gives bulgeDist.

2. **The "cosine formula for triangles" remark.** In the triangle (P, L_r, T) with sides |P − L_r| = d_r (current link), |L_r − T| = linkRestLength (restored link), |P − T| = x (bulge distance along N), and angle at P between the link and the normal with cos θ = dotN_r / d_r, the cosine rule gives linkRestLength² = d_r² + x² − 2·d_r·x·cos θ — the same quadratic. The printed `|L_r|²` (distance from the world origin) appears in no triangle, so it cannot be what "the cosine formula" refers to.

3. **Translation invariance.** bulgeDist is built from dotN_r (link-relative) and must be invariant under rigid translation of the whole system: moving the initial sphere of cells to a different location in world space cannot change the bulging behaviour. `|L_r|²` is not translation-invariant; `|L_r − P|²` is. With `|L_r|²` as printed, a structure far from the origin would drive the radicand negative for essentially every link and the formula would be unusable — the equation only "works" as printed by accident of the figures being simulated near the origin.

4. **Figure 4 corroboration.** In the diagram, each dashed "line of length linkRestLength" runs from L_r to a point on the normal line through P. Those dashed segments close as triangles only if the circle of radius linkRestLength is centred on L_r and intersects the normal line — i.e. the intersection offset from P depends on |L_r − P| (see the detailed Figure 4 description below). Nothing in the diagram references the world origin.

**Related implementation caveat (not discussed in the paper):** the radicand linkRestLength² − |L_r − P|² + dotN_r² goes negative when a link is stretched too far relative to its normal component (d_r² > linkRestLength² + dotN_r²); the paper only claims a bulging effect "when links are in compression". A reimplementation must decide how to clamp the square root for stretched links (e.g. treat the √ term as 0, leaving x_r = dotN_r). This is an implementation decision, not a paper fact.

---

## Figures 1–12: detailed descriptions

All figures are greyscale. Panel counts confirmed by inspection.

**Figure 1** (page 1, left column). *4 panels (2 × 2 grid)*, on white. Four roughly spherical Cellular Forms shown at similar scale, each with a distinct surface character: (top-left) a sphere covered in fine, dense, worm-like (vermicular) convoluted ridges; (top-right) a sphere with much larger, smooth, ruffled lobes/folds; (bottom-left) a sphere of rounded, cracked plates separated by dark crevices (tortoise-shell/brain-like); (bottom-right) a sphere covered in dense, small, granular bumps (berry/shagreen-like).
Caption (verbatim): "**Figure 1.** Examples of Cellular Forms."

**Figure 2** (page 2, left column). *1 panel*, black background. A pale grey, finely granular dendritic structure — a roughly vertical mass that splits into two main branches (the upper branch leaning right), with fuzzy, coral/sponge-like surface texture. This is an artefact of the earlier Aggregation (diffusion-limited aggregation) series.
Caption (verbatim): "**Figure 2.** Example image from the Aggregation series."

**Figure 3** (page 2, right column). *1 panel*, white background. The simulation's initial condition: a ball of smooth, translucent white/grey spheres packed together in a roughly spherical arrangement — a central sphere surrounded by rings of neighbours (≈19 spheres visible; count approximate) with soft shading.
Caption (verbatim): "**Figure 3.** Initial ball of cells, with particles distributed uniformly on the surface of a sphere."

**Figure 4** (page 3, left column) — the bulge-target diagram. *1 panel*, schematic line drawing. Precise geometric description of what is drawn:
- A bold label **N** at the top. From it, a straight line runs vertically down the figure, terminating in an arrowhead just below the N label (the line thus reads as an arrow pointing up toward N — the surface-normal direction).
- Two small **open (unfilled) circles** are centred on this vertical line, one above the other, both above the current particle.
- A small **× (cross) mark** sits on the vertical line exactly midway between the two open circles.
- The bold label **bulgeTarget** stands to the left, with a straight arrow whose tip lands on the **×** mark.
- Below the ×, the vertical line continues down to a **filled black dot labelled P** (the current particle).
- Two further filled black dots: **L₁** at lower left and **L₂** at the right.
- **Solid lines** join P–L₁ and P–L₂ (the two current links, drawn shorter than the rest length, i.e. in compression).
- **Dashed lines**: one from L₁ up to the **lower** open circle, one from L₂ up to the **upper** open circle. The label "Line of length linkRestLength" at the left has a straight arrow pointing at the dashed line from L₁.
Caption (verbatim): "**Figure 4.** Diagram illustrating calculation of the bulge target position."

Geometric interpretation (consistent with equation 4 and its verification): each open circle marks the point on the outward normal ray at which link *r* alone would be restored to its rest length — the outward intersection of the normal ray from P with the sphere of radius `linkRestLength` centred at L_r, at signed distance dotN_r + √(linkRestLength² − |L_r − P|² + dotN_r²) from P. Each dashed line therefore has exactly the length `linkRestLength`, as labelled. The × midway between the two circles — where the bulgeTarget arrow points — is the **averaged** bulge target P + bulgeDist·N (drawn midway for the two-link case, matching the 1/n averaging of the per-link distances). An alternative reading treats the two open circles as the two intersection points of the two rest-length spheres around L₁ and L₂, in which case the outward/upper intersection is "the one used"; note, however, that the dashed lines as printed attach to *different* circles (L₂ → upper, L₁ → lower), which is why the per-link-target + average reading fits the drawing best. Both readings agree on the operative facts: the bulge target lies on the outward normal from P, and the dashed lines represent links restored to exactly `linkRestLength`.

**Figure 5** (page 3, right column, bottom). *6 panels (2 rows × 3 columns)*, white background. Progression of a simulation WITHOUT repulsion: from the initial packed ball of cells (top-left panel), through progressively more crumpled and folded states, to heavily folded, self-overlapping tangles (bottom row) — the surface collapsing into an incoherent mass.
Caption (verbatim): "**Figure 5.** Progression of a form without any repulsive influences between close cells. Structure degenerates into incoherent state."

**Figure 6** (page 4, left column, top). *6 panels (2 rows × 3 columns)*, white background. Same progression WITH the repulsive influence: the initial ball develops folds and creases (including a strongly squashed, creased state at the end of the top row) but recovers into a coherent, rounded, lumpy form — no degeneracy.
Caption (verbatim): "**Figure 6.** Progression of a form with the same parameters but with repulsive influence between close cells. Structure develops coherent shape."

**Figure 7** (page 4, right column, top) — the cell-division diagram. *4 panels in a 2 × 2 grid, each with its own numbered sub-caption; solid lines = established links, dashed lines = chosen/new links, dashed circles = highlighted (selected/new) cells, short radiating "stub" segments on perimeter particles = links continuing to cells outside the illustrated neighbourhood (present in all panels).*

- **Panel 1** (sub-caption verbatim: "1. Parent cell selected and links chosen for plane of cleavage"). 8 filled particles: top-left (TL), top-right (TR), mid-left (ML), centre (C), mid-right (MR), bottom-left (BL), bottom-centre (BC), bottom-right (BR); 14 links total. The centre particle C is ringed by a **dashed circle** (selection marker) and has 7 links. **Exactly two** of C's links are drawn **dashed**: C–TL (up-left) and C–BC (straight down) — these are the two links chosen to represent the plane of cleavage. All other links solid.
- **Panel 2** (sub-caption verbatim: "2. Daughter cell created and some links with parent broken"). A **new daughter particle** appears up-and-right of the parent, ringed by a **dashed circle** and with **no links** (isolated). The parent keeps its dashed circle. The parent has lost 4 links — those on the daughter's side of the cleavage plane (C–TL, C–TR, C–MR, C–BR) — and now retains only 3 solid links (C–ML, C–BL, C–BC). 10 links remain; the network is temporarily disconnected (daughter isolated).
- **Panel 3** (sub-caption verbatim: "3. New links made to daughter cell"). The daughter gains **6 new links, drawn dashed**: to TL, TR, MR, BR, BC, and to the parent itself (a short dashed parent–daughter link). The parent keeps its 3 solid links (ML, BL, BC). Note the bottom-centre neighbour BC is now linked to **both** parent and daughter — this is the "along the plane of cleavage links are made to both the parent and daughter cells" case.
- **Panel 4** (sub-caption verbatim: "4. Final resulting new set of cells and links"). All circles removed; all 16 links solid. 9 particles: parent (degree 4: ML, BL, BC, daughter), daughter (degree 6: TL, TR, MR, BR, BC, parent), and the 7 neighbours. Net effect of the division: **+1 particle, +2 links** (4 parent links broken, 6 daughter links added).
Figure caption (verbatim): "**Figure 7.** The process of cell division."

(Consistency note: the top-left neighbour TL was one of the two *chosen* cleavage links, yet as drawn its parent link is broken and replaced by a daughter link, while the other chosen link BC ends up doubly linked. The illustration is schematic; the text's rule is authoritative.)

**Figure 8** (page 5, right column). *1 panel*, white background. A roughly spherical Cellular Form rendered in the solid-surface style: many smooth, lobed "plates" separated by dark crevices, shaded with ambient occlusion (soft self-shadowing), giving a sculptural studio-lit appearance.
Caption (verbatim): "**Figure 8.** Cellular Form rendered using solid surface and ambient occlusion shading."

**Figure 9** (page 6, left column, top). *1 panel*, black background. The **same simulation data as Figure 8** re-rendered in the X-ray style: a ghostly, translucent white/grey density accumulation forming a spherical envelope, with fine radial filamentary wisps and brighter knots toward the centre; internal structure invisible in the solid render is revealed.
Caption (verbatim): "**Figure 9.** The same simulation data as Figure 8 rendered using an X-Ray style density accumulation map."

**Figure 10** (page 6, right column, full height). *5 panels stacked vertically*, showing the diversity of surface patterns/shapes: (1) a dense field of smooth, rounded, pebble-like cells of similar size, tightly packed; (2) larger rounded lobes separated by fine reticulated crevices/veins (brain-coral-like); (3) convoluted, worm-like tangled ridges (brain-like folding); (4) a dark, dramatically lit image of sharp blade/spike-like protrusions (plant-like pointed shards on black); (5) a dense, irregular, lumpy granular surface (moss/coral-like fine bumps).
Caption (verbatim): "**Figure 10.** Examples of different surface patterns and shapes on Cellular Forms."

**Figure 11** (page 7, left column). *3 panels side by side*, white background. Three upright, cone/teardrop-shaped plant-like forms, each tapering to a point at the top and broadest near the base, with fine granular/branching surface texture; the middle form is darker and slightly narrower than the two flanking lighter forms. Produced with nutrient created by light from a single direction and low nutrient diffusion.
Caption (verbatim): "**Figure 11.** Examples of plant-like forms with nutrient created using light from a single direction."

**Figure 12** (page 7, left column). *1 panel*, white background. A large, roughly spherical Cellular Form covered in rounded, worm-like convoluted lobes (brain-like surface) — exhibited as evidence that reaction-diffusion-like patterns emerge without any explicit reaction-diffusion mechanism.
Caption (verbatim): "**Figure 12.** Cellular Form with surface shapes reminiscent of patterns formed by reaction-diffusion equations."

---

## Table 1 (verbatim transcription)

Caption (verbatim): "**Table 1.** Class structure used for particle data."

| Class name | Class description |
|---|---|
| BaseParticleGpu | Base class defining particle data to be held on the GPU. |
| LinkedParticleGpu | Derived from BaseParticleGpu. Extends the base class by adding a list of links between particles. Also implements support for reaction-diffusion equations using the links between particles. |
| ElasticSheetParticleGpu | Derived from LinkedParticleGpu. Adds methods for dealing with the spring, planar and bulge effects, as well as repulsion effects between spatially close particles. |
| FoodSplitParticleGpu | Derived from ElasticSheetParticleGPU. Adds methods to implement food levels, nutrient levels, and cell division. |

(Verified at 3× zoom. Note the printed inconsistency in the last row's description: "ElasticSheetParticle**GPU**" with all-capitals GPU, whereas the class name column and the previous row use "ElasticSheetParticle**Gpu**". Transcribed as printed.)

---

## Footnotes

Exactly one footnote in the paper, keyed to the author name "Andy Lomas¹" on page 1 and set at the foot of the first column beneath a short horizontal rule (not at the foot of the page):

> ¹Website: www.andylomas.com, Email: andylomas@yahoo.com

(Verified at 4× zoom.)

---

## References (verbatim, complete)

> [1] Wikipedia article on Systems Art: http://en.wikipedia.org/wiki/Systems_art
> [2] E. Haeckel, *Art Forms in Nature*, Dover Publications Inc.; Revised edition (2 Jan 2000).
> [3] D.W. Thompson, *On Growth and Form*, Cambridge University Press; New edition (31 July 1992).
> [4] A.M. Turing, "*The Chemical Basis of Morphogenesis*", Philosophical Transactions of the Royal Society of London 237 (641), 37-72.
> [5] G. Turk, "*Generating textures on arbitrary surfaces using reaction-diffusion*", Proceedings of SIGGRAPH '91, 289-298.
> [6] A. Lindenmayer, "*Developmental systems without cellular interaction, their languages and grammars*". Journal of Theoretical Biology, 30 (1971), 455-484.
> [7] T.A. Witten Jr & L.M. Sander, "*Diffusion-Limited Aggregation, a Kinetic Critical Phenomenon*", Phys. Rev. Lett. 47, 1400 (1981).
> [8] A. Lomas, "*Aggregation: Complexity out of Simplicity*", sketch session SIGGRAPH 2005, http://www.andylomas.com/sketch_0087_final.pdf
> [9] A. Lomas, video: "*Growth by Aggregation*", https://vimeo.com/83297099
> [10] J.A. Kaandorp, "*Analysis and synthesis of radiative accretive growth in three dimensions*", Journal of Theoretical Biology, 175 (1995), 39-55.
> [11] J.A. Kaandorp & J. Kübler, *The Algorithmic Beauty of Seaweeds, Sponges, and Corals*, Springer-Verlag (2001), 125-144.
> [12] G. Hart, "*Growth Forms*", Proceedings of Bridges 2009, http://georgehart.com/Growth/GrowthForms.pdf
> [13] W.T. Reeves, "*Particle Systems: A Technique for Modelling a Class of Fuzzy Object*", ACM Transactions on Graphics, Volume 2 Issue 2, April 1983, 91-108.
> [14] CUDA http://www.nvidia.com/object/cuda_home_new.html
> [15] A. Lomas, video: "*Cellular Forms*", https://vimeo.com/82989945
> [16] A. Lomas, video: "*Cellular Forms (X_Ray version)*", https://vimeo.com/83294152
> [17] Video: "*The Development of a Frog*", http://www.youtube.com/watch?v=dXpAbezdOho
> [18] Video: "*Early mitotic division waves in Drosophila embryo*", http://www.youtube.com/watch?v=--NfCyWNodA
> [19] Wikipedia article on Voronoi diagrams: http://en.wikipedia.org/wiki/Voronoi_diagram
> [20] A. Lomas, video: "*Cellular Form Generation Test 003c5*", http://www.youtube.com/watch?v=kcEBEIAhsuk

Placement note: the "REFERENCES" heading and reference [1] appear at the bottom of page 7's right column; page 8 contains only references [2]–[20], set as a single narrow column. Reference [3] is printed "D.W. Thompson" (the body text uses "D'Arcy Thompson's"); reference [13]'s title ends "Fuzzy Object" as far as can be resolved at available resolution (the original Reeves paper title has "Fuzzy Objects").

---

## Implementation facts (every quantitative fact in the paper)

**Simulation model**
- Cell = one particle; links = direct attachments; topology of connected particles is that of a two-dimensional surface (simple closed surface, topologically equivalent to a sphere — no holes, not a solid volume).
- Initial condition for all illustrated structures: cells uniformly spaced on the surface of a sphere (a "simple ball of cells").
- Time advances in uniform clock-cycle steps; no momentum term in the update (viscous-medium justification).
- Simulation parameters: `linkRestLength`, `springFactor`, `planarFactor`, `bulgeFactor`; repulsion parameters: `roi` (radius of influence), `repulsionStrength`.
- Cell division triggered when internal 'food level' exceeds a (unspecified) threshold; division changes topology per the Figure 7 process (two chosen cleavage links; side links re-assigned; cleavage-plane neighbours linked to both; new parent–daughter link).
- Food/growth methods implemented: (1) uniform random increment per cell per time step; (2) reaction-diffusion equations over the surface, chemicals stored at cells and diffused along direct links, one chemical acting as 'nutrient level' scaling the food increment; (3) ray-traced light from outside the structure stimulating nutrient creation in hit cells, nutrient then diffusing cell-to-cell along links.
- All figures/forms in the paper were created WITHOUT RDEs — only uniform growth rates, or light-created nutrient that diffuses through the structure.
- Observation: low nutrient diffusion + single-direction light → plant-like forms; uniform growth (or high nutrient diffusion) → internal-organ-like (brain-like) folded forms.

**Scale & hardware**
- Typical simulations: tens of thousands of iterations; datasets grow to over 50 million particles.
- Quick parameter-sampling tests: maximum of 1 million cells, typically 1–5 minutes per sample run.
- Full-length simulations: typically 52–56 million cells, stopping at the memory limit of the current hardware — **NVIDIA GTX Titan with 6GB RAM**; typically 1–4 hours each.
- Previous related work (Kaandorp, Hart): simulations run for only a few thousand cell primitives.
- Implementation language: C++ with CUDA [14]; all per-particle computation (forces, ray casting for rendering/light, topology changes on division) runs on the GPU; all cell data kept purely in GPU memory; CPU↔GPU transfers only for disk file I/O; the earlier CPU implementation of the repulsion term was too slow to reach the required cell counts.
- Software is an extensible object-oriented framework with the 4-class hierarchy of Table 1 (BaseParticleGpu → LinkedParticleGpu → ElasticSheetParticleGpu → FoodSplitParticleGpu).
- Future-work motivation: GPU memory is the main limit on structure complexity; out-of-core or multi-GPU distribution would remove/relax it (noted as possibly having impractical performance implications).

**Rendering**
- Ray-tracing; cells treated as spheres with radii based on the average distance to linked cells; all key rendering functions are CUDA kernels on the GPU.
- Style 1 — solid surface with 'ambient occlusion' self-shadowing; uniform omni-directional diffuse light on a Lambertian (ideal diffuse) surface; "directionally unbiased" view; deliberately echoes Ernst Haeckel's illustrative technique.
- Style 2 — 'X-Ray': accumulated density along rays through the whole structure (each sphere contributes to an accumulated density), revealing internal structure; likened to biological microscopy.
- Artefact types: (1) high-resolution prints, currently rendered at **8600 × 8600 pixels**; (2) high-definition animations rendered at equally spaced time intervals; (3) 3D stereo views from camera pairs with an interocular separation.
- Voronoi-cell rendering (to illustrate cell walls) was explored in early tests [19], [20] but is not implemented on the GPU.
- 3D printing of the structures proposed as future work.

**Results claims**
- Emergent behaviours: complex folding, branching, rhythmic pulses, waves of growth propagating across surfaces (observed especially in the animations [15], [16]); likened to pulses of growth during embryonic cell division [17] and mitotic division waves [18].
- Without repulsion between close non-linked cells, forms degenerate into a meaningless unstructured mess (surface self-intersects as it folds) — see Figure 5 vs Figure 6.

---

## OCR-risk log (characters/words flagged during extraction)

1. **bulgeDist `|L_r|²`** — printed as the squared norm of the absolute position; must be `|L_r − P|²` (squared current link length). See "Formula verification". Resolved (typo), not an OCR issue.
2. **"differntiation"** (page 4, first simplification bullet) — printed exactly so, missing an "e" (should be "differentiation"). Confirmed at 3×; flagged [sic].
3. **"tracing a ray though the whole structure"** (page 5) — printed "though" for "through". Confirmed at 4×; flagged [sic].
4. **"each of these target position"** (page 3, before the P′ equation) — trailing "s" (positions vs position) not resolvable at available resolution.
5. **Reference [3]** — printed "D.W. Thompson"; body text says "D'Arcy Thompson's". Transcribed as printed.
6. **Reference [12] URL** — "http://georgehart.com/Growth/GrowthForms.pdf"; the "Growth/" path segment read at 4× but is unusual (could conceivably be "GrowthForms/").
7. **Reference [8] filename** — "sketch_0087_final.pdf"; digits read at moderate confidence.
8. **Reference [13]** — title ends "…Class of Fuzzy Object" as resolved; a trailing "s" cannot be ruled out at this resolution (original paper: "Fuzzy Objects").
9. **Reference [16]** — "(X_Ray version)": underscore vs hyphen ("X-Ray") not fully resolvable.
10. **Reference [14] URL** — "cuda_home_new.html": underscores vs spaces not fully resolvable at this resolution (printed with underline styling).
11. **"omnidirectional"** (page 5, §5) vs **"omni-directional"** (page 6, §6) — the hyphen difference between the two occurrences is as printed (each verified in context).
12. **Figure 4 semantics** — the drawing shows the L₂ dashed line attaching to the *upper* open circle and the L₁ dashed line to the *lower* open circle, with the × (where the bulgeTarget arrow points) exactly midway between them; the per-link-target + average reading is preferred (see Figure 4 description), but the two-circle symbolism admits the alternative "two sphere-intersection points" reading. Operative geometry is identical either way.
13. **Figure 7 link bookkeeping** — panel-level link counts (8 particles/14 links → 10 links after break → 16 links, 9 particles at the end) are confident at 2× zoom; the identity of which specific neighbour sits "on the daughter's side" is a schematic choice in the drawing (TL ends up daughter-only despite being a chosen cleavage link).
14. **"inter-cellular diffusion rate for nutrient diffusion rates are high"** (page 6) — grammatically awkward but verified verbatim at 4×.
15. **Abstract "strong reminiscences of plants"** — verified at half-page zoom; not "resemblances".
16. **Related Work "particularly following the work of Greg Turk [5]"** — verified at half-page zoom.
