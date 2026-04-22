# Shared workflow: imported by babbitt_pygslib_tutorial.jl and the IJulia notebook.
# After `include`, call `main(; tutorial_project_dir=...)` for notebooks.

using CSV
using DataFrames
using DrillHoles
using GeoIO
using GeoStats
using GeoStatsFunctions: Variogram, fit
using Meshes
using Random
using Unitful
import CairoMakie

function babbitt_data_candidates(tutorial_project_dir::AbstractString=String(@__DIR__))
    tut = abspath(tutorial_project_dir)
    teaching = joinpath(tut, "..", "..", "..", "..")
    pygslib_2023 = joinpath(teaching, "Teaching 2023", "MINE 420 - 2023", "Scripts and Data", "PyGSLIB_Tutorial1")
    julia_2025 = joinpath(dirname(tut), "..", "Resource estimation in Julia", "data")
    unique!(filter(isdir, [pygslib_2023, julia_2025]))
end

function find_babbitt_data_dir(; tutorial_project_dir::AbstractString=String(@__DIR__))
    if haskey(ENV, "BABBITT_DATA")
        p = abspath(ENV["BABBITT_DATA"])
        isdir(p) || error("BABBITT_DATA is not a directory: $p")
        return p
    end
    collar = "collar_BABBITT.csv"
    for d in babbitt_data_candidates(tutorial_project_dir)
        isfile(joinpath(d, collar)) && return abspath(d)
    end
    error("""
    Could not find Babbitt CSV tables. Set for example:
      ENV["BABBITT_DATA"] = raw"/path/to/PyGSLIB_Tutorial1"
    (folder must contain collar_BABBITT.csv, survey_BABBITT.csv, assay_BABBITT.csv, and domain.stl / Mpz.stl / MTZ.stl)
    """)
end

function find_domain_stl(data_dir::AbstractString)
    for name in ("domain.stl", "Mpz.stl", "MTZ.stl")
        p = joinpath(data_dir, name)
        isfile(p) && return p
    end
    error("No domain.stl, Mpz.stl, or MTZ.stl in: $data_dir")
end

const XORG = 2288230.0
const YORG = 415200.0
const ZORG = -1000.0
const DX, DY, DZ = 100.0, 100.0, 30.0
const NX, NY, NZ = 160, 100, 90

const BLOCK_STRIDE = parse(Int, get(ENV, "BLOCK_STRIDE", "4"))
const VARIO_MAX_SAMPLES = parse(Int, get(ENV, "VARIO_MAX_SAMPLES", "8000"))

function ensure_string_bhid!(df::DataFrame, col=:BHID)
    df[!, col] = string.(df[!, col])
    df
end

function composite_block_centroids(stride::Int)
    ft = u"ft"
    orig = Point((XORG * ft, YORG * ft, ZORG * ft))
    spac = (DX * ft, DY * ft, DZ * ft)
    grid = CartesianGrid(orig, spac, GridTopology(NX, NY, NZ))
    topo = topology(grid)
    vec([
        centroid(grid, cart2elem(topo, i, j, k))
        for i in 1:stride:NX, j in 1:stride:NY, k in 1:stride:NZ
    ])
end

function main(; tutorial_project_dir::AbstractString=String(@__DIR__))
    data_dir = find_babbitt_data_dir(; tutorial_project_dir)
    collar_path = joinpath(data_dir, "collar_BABBITT.csv")
    survey_path = joinpath(data_dir, "survey_BABBITT.csv")
    assay_path = joinpath(data_dir, "assay_BABBITT.csv")
    stl_path = find_domain_stl(data_dir)

    for (label, p) in [("collar", collar_path), ("survey", survey_path), ("assay", assay_path), ("STL domain", stl_path)]
        isfile(p) || error("Missing $label file: $p\nSet ENV[\"BABBITT_DATA\"] or use PyGSLIB_Tutorial1 next to the course layout.")
    end

    println("Data directory: ", data_dir)
    println("STL file: ", basename(stl_path))

    collar_df = CSV.read(collar_path, DataFrame)
    survey_df = CSV.read(survey_path, DataFrame)
    assay_df = CSV.read(assay_path, DataFrame)

    ensure_string_bhid!(collar_df, :BHID)
    ensure_string_bhid!(survey_df, :BHID)
    ensure_string_bhid!(assay_df, :BHID)

    cu_sym = hasproperty(assay_df, :CU) ? :CU : :Cu
    assay_df = select(assay_df, :BHID, :FROM, :TO, cu_sym)
    rename!(assay_df, cu_sym => :CU)
    assay_df.CU = coalesce.(assay_df.CU, 0.0)

    collar = Collar(collar_df; holeid=:BHID, x=:XCOLLAR, y=:YCOLLAR, z=:ZCOLLAR)
    survey = Survey(survey_df; holeid=:BHID, at=:AT, azm=:AZ, dip=:DIP)
    assay = Interval(assay_df; holeid=:BHID, from=:FROM, to=:TO)

    ft = u"ft"
    holes_tab = desurvey(collar, survey, [assay]; len=10.0ft, inunit=ft, outunit=ft, geom=:none)
    pts = PointSet([Point((r.X, r.Y, r.Z)) for r in eachrow(holes_tab)])
    samples = georef(DataFrame(HOLEID=holes_tab.HOLEID, FROM=holes_tab.FROM, TO=holes_tab.TO, CU=holes_tab.CU), pts)
    println("Desurveyed composite samples: ", nelements(domain(samples)))

    stl_gt = GeoIO.load(stl_path; lenunit=ft, repair=true)
    ore_shell = domain(stl_gt)

    dom = domain(samples)
    inside = [sideof(centroid(dom, i), ore_shell) == IN for i in 1:length(dom)]
    samples_ore = samples[inside, :]
    println("Samples inside STL: ", nelements(domain(samples_ore)))

    nelements(domain(samples_ore)) == 0 && error("No composites inside the solid; check STL lenunit=ft and coordinate alignment.")

    centers = composite_block_centroids(BLOCK_STRIDE)

    in_blk = map(c -> sideof(c, ore_shell) == IN, centers)
    block_pts = centers[in_blk]
    println("Strided block centroids inside STL: ", length(block_pts), " (stride=$BLOCK_STRIDE)")

    dom_o = domain(samples_ore)
    df_cu = DataFrame(Cu=Float64.(values(samples_ore).CU))
    sdata = georef(df_cu, dom_o)

    rng = MersenneTwister(42)
    nsub = min(VARIO_MAX_SAMPLES, nelements(domain(sdata)))
    subix = sort(shuffle(rng, 1:nelements(domain(sdata)))[1:nsub])
    ssub = sdata[subix, :]

    maxlag = 850.0ft
    g = EmpiricalVariogram(ssub, :Cu, maxlag=maxlag)
    γ = fit(Variogram, g, h -> 1 / h^2)
    println("Fitted variogram: ", γ)

    tgt = PointSet(block_pts)
    estim = sdata |> Select(:Cu) |> InterpolateNeighbors(tgt, model=Kriging(γ))
    cu_est = Float64.(estim.Cu)
    println("OK mean Cu (%): ", sum(cu_est) / length(cu_est))
    println("OK max Cu (%): ", maximum(cu_est))

    zmid = (ZORG + 0.5 * NZ * DZ) * ft
    tol = 20.0ft
    slice_i = findall(block_pts) do p
        abs(to(p)[3] - zmid) < tol
    end
    if !isempty(slice_i)
        fig = CairoMakie.Figure(size=(720, 600))
        ax = CairoMakie.Axis(fig[1, 1]; aspect=CairoMakie.DataAspect(), xlabel="X (ft)", ylabel="Y (ft)",
            title="Ordinary kriging Cu (%) — mid-Z slice (stride=$BLOCK_STRIDE)")
        pts = block_pts[slice_i]
        vals = cu_est[slice_i]
        xs = [ustrip(u"ft", to(p)[1]) for p in pts]
        ys = [ustrip(u"ft", to(p)[2]) for p in pts]
        hi = max(0.5, quantile(vals, 0.98))
        CairoMakie.scatter!(ax, xs, ys; color=vals, colormap=:inferno, markersize=8, colorrange=(0.0, hi))
        CairoMakie.Colorbar(fig[1, 2]; label="Cu (%)", colormap=:inferno, colorrange=(0.0, hi))
        out_png = joinpath(abspath(tutorial_project_dir), "babbitt_ok_plan_slice.png")
        CairoMakie.save(out_png, fig)
        println("Wrote ", out_png)
    end

    println("\nDone. Set ENV[\"BLOCK_STRIDE\"]=\"1\" for full block density (slower).")

    return (; samples_ore, estim, γ, block_pts)
end
