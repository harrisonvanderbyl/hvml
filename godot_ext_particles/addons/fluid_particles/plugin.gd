@tool
extends EditorPlugin

# ─── FluidParticleSystem editor plugin ───────────────────────────────────────
# Drop this addon into res://addons/fluid_particles/ of any Godot 4.3+ project.
# Provides:
#   • Bounding-box gizmo drawn in the 3-D viewport
#   • Inspector-editable shader paths with defaults pointing to addon shaders
#   • Toolbar debug-mode selector
# ─────────────────────────────────────────────────────────────────────────────

const ADDON_BASE   := "res://addons/fluid_particles"
const SHADER_DIR   := ADDON_BASE + "/shaders"

var _gizmo_plugin: FluidGizmoPlugin = null
var _toolbar: HBoxContainer         = null
var _mode_btn: OptionButton         = null

func _enter_tree() -> void:
	_gizmo_plugin = FluidGizmoPlugin.new()
	add_node_3d_gizmo_plugin(_gizmo_plugin)

	_toolbar = HBoxContainer.new()
	var lbl := Label.new()
	lbl.text = "Fluid Debug: "
	_toolbar.add_child(lbl)

	_mode_btn = OptionButton.new()
	_mode_btn.add_item("None",             0)
	_mode_btn.add_item("Bounding Box",     1)
	_mode_btn.add_item("Simple Points",    2)
	_mode_btn.add_item("Static (no sim)",  3)
	_mode_btn.connect("item_selected", _on_mode_selected)
	_toolbar.add_child(_mode_btn)

	add_control_to_container(EditorPlugin.CONTAINER_SPATIAL_EDITOR_MENU, _toolbar)

func _exit_tree() -> void:
	remove_node_3d_gizmo_plugin(_gizmo_plugin)
	if _toolbar:
		remove_control_from_container(EditorPlugin.CONTAINER_SPATIAL_EDITOR_MENU, _toolbar)
		_toolbar.queue_free()

# ── Toolbar ───────────────────────────────────────────────────────────────────
func _on_mode_selected(idx: int) -> void:
	var root := get_editor_interface().get_edited_scene_root()
	if root:
		_apply_mode_recursive(root, idx)

func _apply_mode_recursive(node: Node, mode: int) -> void:
	if node.get_class() == "FluidParticleSystem":
		node.set("debug_mode", mode)
	for child in node.get_children():
		_apply_mode_recursive(child, mode)

# ── Node creation helper ──────────────────────────────────────────────────────
# Called when the user instantiates a new FluidParticleSystem from the scene
# editor — fills in default shader paths from the addon.
func _make_visible(visible: bool) -> void:
	pass  # nothing extra to show/hide

func _handles(object: Object) -> bool:
	return object != null and object.get_class() == "FluidParticleSystem"

func _edit(object: Object) -> void:
	if _gizmo_plugin and object:
		update_overlays()

# ─────────────────────────────────────────────────────────────────────────────
class FluidGizmoPlugin extends EditorNode3DGizmoPlugin:

	func _get_gizmo_name() -> String:
		return "FluidParticleSystem"

	func _has_gizmo(node: Node3D) -> bool:
		return node.get_class() == "FluidParticleSystem"

	func _init() -> void:
		create_material("bb_wire", Color(0.2, 1.0, 0.4, 0.9), false, true)

	func _redraw(gizmo: EditorNode3DGizmo) -> void:
		gizmo.clear()
		var node: Node3D = gizmo.get_node_3d()
		if not node.has_method("get_grid_aabb"):
			return

		var aabb: AABB = node.call("get_grid_aabb")
		var p := aabb.position
		var s := aabb.size
		var c := [
			p,
			p + Vector3(s.x, 0,   0  ),
			p + Vector3(s.x, s.y, 0  ),
			p + Vector3(0,   s.y, 0  ),
			p + Vector3(0,   0,   s.z),
			p + Vector3(s.x, 0,   s.z),
			p + Vector3(s.x, s.y, s.z),
			p + Vector3(0,   s.y, s.z),
		]
		var edges := [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
		var lines := PackedVector3Array()
		for e in edges:
			lines.append(c[e[0]])
			lines.append(c[e[1]])

		gizmo.add_lines(lines, get_material("bb_wire", gizmo))
