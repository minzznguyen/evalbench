import os
import mesop as me
import pandas as pd
import yaml
import logging
import json

logging.basicConfig(level=logging.INFO)

# Manually enable debug mode to bypass XSRF check if needed
# (e.g. when running in container behind a proxy)
if os.environ.get("MESOP_XSRF_CHECK") == "false":
    try:
        from mesop.runtime import runtime
        runtime().debug_mode = True
    except Exception as e:
        logging.error(f"Failed to enable debug mode: {e}")

try:
    import dashboard
    import conversations
except ImportError:
    # Optional modules could not be imported; continue without them.
    logging.warning(
        "Optional modules 'dashboard', and 'conversations' "
        "could not be imported (absolute or relative)."
    )


def df_to_config(df: pd.DataFrame) -> dict:
    import ast

    original_dict = {}

    for _, row in df.iterrows():
        key_path = row["config"]
        value_str = row["value"]

        try:
            if pd.isna(value_str):
                value = None
            else:
                value = ast.literal_eval(value_str)
        except (ValueError, SyntaxError, TypeError):
            value = value_str

        keys = key_path.split(".")

        current_level = original_dict
        for key in keys[:-1]:
            if key not in current_level:
                current_level[key] = {}
            current_level = current_level[key]

        current_level[keys[-1]] = value

    return original_dict


@me.stateclass
class State:
    selected_directory: str
    selected_tab: str = "Dashboard"
    conversation_index: int = 0
    eval_summaries: str = ""
    eval_id_filter: str = ""
    product_filter: str = ""
    requester_filter: str = ""
    sort_column: str = "date"
    sort_descending: bool = True
    open_dropdown: str = ""


def get_results_dir():
    # Check multiple locations for results directory
    results_dir_candidates = [
        "/tmp_session_files/results",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "results"),
        os.path.join(os.getcwd(), "results"),
    ]

    for candidate in results_dir_candidates:
        if os.path.exists(candidate) and os.path.isdir(candidate):
            return candidate

    return results_dir_candidates[1]  # Fallback to default


def get_eval_details(results_dir, dir_name):
    details = {
        "product": "N/A",
        "date": "N/A",
        "requester": "N/A",
        "exact_match": "N/A",
        "llmrater": "N/A",
        "trajectory_matcher": "N/A",
        "turn_count": "N/A",
        "executable": "N/A",
        "token_consumption": "N/A",
        "end_to_end_latency": "N/A",
    }

    # Get product
    config_path = os.path.join(results_dir, dir_name, "configs.csv")
    if os.path.exists(config_path):
        try:
            df = pd.read_csv(config_path)
            # Check for both typo and correct spelling
            row = df[
                df["config"].isin(
                    [
                        "experiment_config.poduct_name",
                        "experiment_config.product_name",
                    ]
                )
            ]
            if not row.empty:
                details["product"] = str(row["value"].iloc[0])

            # Check for requester
            req_row = df[
                df["config"].isin(
                    [
                        "experiment_config.experiment_config.guitar_requester",
                        "experiment_config.guitar_requester",
                    ]
                )
            ]
            if not req_row.empty:
                details["requester"] = str(req_row["value"].iloc[0])
        except Exception as e:
            logging.warning(f"Error reading configs.csv for {dir_name}: {e}")

    # Get summary metrics
    summary_path = os.path.join(results_dir, dir_name, "summary.csv")
    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)
            if "run_time" in df.columns and not df.empty:
                details["date"] = str(df["run_time"].iloc[0])
            for _, row in df.iterrows():
                name = row.get("metric_name")
                correct = row.get("correct_results_count", 0)
                total = row.get("total_results_count", 0)
                pct = (correct / total) * 100 if total > 0 else 0
                if name == "exact_match":
                    details["exact_match"] = f"{pct:.0f}%"
                elif name == "llmrater":
                    details["llmrater"] = f"{pct:.0f}%"
                elif name == "trajectory_matcher":
                    details["trajectory_matcher"] = f"{pct:.0f}%"
                elif name == "turn_count":
                    details["turn_count"] = f"{correct:.1f}"
                elif name == "executable":
                    details["executable"] = f"{pct:.0f}%"
                elif name == "token_consumption":
                    details["token_consumption"] = f"{correct:.0f}"
                elif name == "end_to_end_latency":
                    details["end_to_end_latency"] = f"{correct:.0f}"
        except Exception as e:
            logging.warning(f"Error reading summary.csv for {dir_name}: {e}")

    return details


def get_color_for_pct(val_str):
    if not val_str or not val_str.endswith("%"):
        return "#334155"  # Default color
    try:
        val = float(val_str.rstrip("%"))
        if val >= 80:
            return "#16a34a"  # Green
        elif val >= 40:
            return "#ca8a04"  # Yellow
        else:
            return "#dc2626"  # Red
    except Exception:
        return "#334155"


def on_load(e: me.LoadEvent):
    state = me.state(State)
    results_dir = get_results_dir()
    directories = []
    if os.path.exists(results_dir):
        # List directories only
        directories = [
            d
            for d in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, d))
        ]

    job_id = me.query_params.get("job_id") or me.query_params.get("jobid")
    if job_id and job_id in directories:
        state.selected_directory = job_id


@me.page(
    path="/",
    title="EvalBench Viewer",
    on_load=on_load,
    security_policy=me.SecurityPolicy(
        dangerously_disable_trusted_types=True,
        cross_origin_opener_policy="same-origin",
    ),
    stylesheets=[
        "data:",
        "/static/custom.css",
    ],
)
def app():
    with me.box(
        style=me.Style(
            background="#f8fafc",
            min_height="100vh",
            width="100%",
        )
    ):
        render_app_content()


def render_app_content():
    state = me.state(State)
    results_dir = get_results_dir()

    directories = []
    if os.path.exists(results_dir):
        # List directories only
        directories = [
            d
            for d in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, d))
        ]

    def on_title_click(e: me.ClickEvent):
        state.selected_directory = ""
        state.conversation_index = 0

    # Full-width header bar
    with me.box(
        style=me.Style(
            background="#1e293b",
            padding=me.Padding.symmetric(vertical="16px", horizontal="5%"),
            margin=me.Margin(bottom="24px"),
            display="flex",
            justify_content="space-between",
            align_items="center",
        )
    ):
        me.button(
            "EvalBench Viewer",
            on_click=on_title_click,
            style=me.Style(
                color="#f8fafc",
                font_size="22px",
                font_weight="700",
                letter_spacing="0.5px",
                background="transparent",
                padding=me.Padding.all("0px"),
                margin=me.Margin.all("0px"),
                border=me.Border.all(me.BorderSide(width="0px")),
                text_align="left",
            ),
        )

    # Centered content at 90% browser width
    with me.box(
        style=me.Style(
            width="90%",
            margin=me.Margin.symmetric(horizontal="auto"),
            display="flex",
            flex_direction="column",
            gap="16px",
            background="#ffffff",
            padding=me.Padding.all("24px"),
            border_radius="8px",
            box_shadow="0 4px 6px -1px rgba(0, 0, 0, 0.1)",
            color="#1e293b",
        )
    ):

        if state.selected_directory:

            def on_tab_change(e: me.ButtonToggleChangeEvent):
                state.selected_tab = e.value

            me.button_toggle(
                value=state.selected_tab,
                buttons=[
                    me.ButtonToggleButton(
                        label="Dashboard", value="Dashboard"
                    ),
                    me.ButtonToggleButton(
                        label="Configs", value="Configs"
                    ),
                    # me.ButtonToggleButton(label="Evals", value="Evals"),
                    # me.ButtonToggleButton(label="Scores", value="Scores"),
                    me.ButtonToggleButton(
                        label="Conversations", value="Conversations"
                    ),
                    # me.ButtonToggleButton(label="Summary", value="Summary"),
                ],
                on_change=on_tab_change,
            )

            if state.selected_tab == "Dashboard":
                dashboard.dashboard_component(
                    os.path.join(results_dir, state.selected_directory)
                )
            elif state.selected_tab == "Conversations":

                def on_prev_conversation(e: me.ClickEvent):
                    s = me.state(State)
                    if s.conversation_index > 0:
                        s.conversation_index -= 1

                def on_next_conversation(e: me.ClickEvent):
                    s = me.state(State)
                    s.conversation_index += 1

                conversations.conversations_component(
                    os.path.join(results_dir, state.selected_directory),
                    conversation_index=state.conversation_index,
                    on_prev=on_prev_conversation,
                    on_next=on_next_conversation,
                )
            elif state.selected_tab == "Configs":
                config_path = os.path.join(
                    results_dir, state.selected_directory, "configs.csv"
                )
                if os.path.exists(config_path):
                    try:
                        df = pd.read_csv(config_path)
                        config = df_to_config(df)
                        me.code(yaml.dump(config))
                    except Exception as e:
                        me.text(f"Error reading configs.csv: {e}")
                else:
                    me.text(
                        f"configs.csv not found in {state.selected_directory}"
                    )
            elif state.selected_tab == "Evals":
                evals_path = os.path.join(
                    results_dir, state.selected_directory, "evals.csv"
                )
                if os.path.exists(evals_path):
                    try:
                        df = pd.read_csv(evals_path)
                        details = get_eval_details(
                            results_dir, state.selected_directory
                        )
                        df.insert(0, "orchestrator", details["orchestrator"])
                        me.table(data_frame=df)
                    except Exception as e:
                        me.text(f"Error reading evals.csv: {e}")
                else:
                    me.text(
                        f"evals.csv not found in {state.selected_directory}"
                    )
            elif state.selected_tab == "Scores":
                scores_path = os.path.join(
                    results_dir, state.selected_directory, "scores.csv"
                )
                if os.path.exists(scores_path):
                    try:
                        df = pd.read_csv(scores_path)
                        me.table(data_frame=df)
                    except Exception as e:
                        me.text(f"Error reading scores.csv: {e}")
                else:
                    me.text(
                        f"scores.csv not found in {state.selected_directory}"
                    )
            elif state.selected_tab == "Summary":
                summary_path = os.path.join(
                    results_dir, state.selected_directory, "summary.csv"
                )
                if os.path.exists(summary_path):
                    try:
                        df = pd.read_csv(summary_path)
                        me.table(data_frame=df)
                    except Exception as e:
                        me.text(f"Error reading summary.csv: {e}")
                else:
                    me.text(
                        f"summary.csv not found in {state.selected_directory}"
                    )
        else:
            with me.box(
                style=me.Style(
                    background="#ffffff",
                    padding=me.Padding.all("24px"),
                    border_radius="12px",
                    border=me.Border.all(
                        me.BorderSide(
                            width="1px",
                            color="#e5e7eb",
                            style="solid",
                        )
                    ),
                    box_shadow="0 1px 3px rgba(0,0,0,0.06)",
                    text_align="center",
                    margin=me.Margin(top="16px"),
                )
            ):
                me.text(
                    "Welcome to EvalBench Viewer",
                    style=me.Style(
                        font_size="24px",
                        font_weight="700",
                        color="#1f2937",
                        margin=me.Margin(bottom="8px"),
                    ),
                )
                me.text(
                    f"Found {len(directories)} evaluation runs. "
                    "Click on an Eval ID in the table below to explore "
                    "the results.",
                    style=me.Style(
                        font_size="16px",
                        color="#6b7280",
                        margin=me.Margin(bottom="16px"),
                    ),
                )
                if directories:
                    # Compute summaries if empty
                    s = me.state(State)
                    summaries = []
                    if s.eval_summaries:
                        try:
                            summaries = json.loads(s.eval_summaries)
                        except Exception:
                            summaries = []

                    if not summaries:
                        for d in sorted(directories):
                            details = get_eval_details(results_dir, d)
                            summaries.append({
                                "id": d,
                                "date": details.get("date", "N/A"),
                                "product": details["product"],
                                "requester": details.get("requester", "N/A"),
                                "exact_match": details["exact_match"],
                                "llmrater": details["llmrater"],
                                "trajectory_matcher": details[
                                    "trajectory_matcher"
                                ],
                                "turn_count": details["turn_count"],
                                "executable": details["executable"],
                                "token_consumption": details[
                                    "token_consumption"
                                ],
                                "end_to_end_latency": details[
                                    "end_to_end_latency"
                                ]
                            })
                        s.eval_summaries = json.dumps(summaries)

                    # Sort by selected column
                    reverse = state.sort_descending
                    col = state.sort_column

                    def get_sort_key(x):
                        val = x.get(col, "N/A")

                        # Handle numbers and percentages
                        if col in [
                            "exact_match",
                            "llmrater",
                            "trajectory_matcher",
                            "executable",
                        ]:
                            if val == "N/A":
                                return -1.0 if reverse else 101.0
                            if val.endswith("%"):
                                try:
                                    return float(val.rstrip("%"))
                                except ValueError:
                                    return -1.0 if reverse else 101.0
                            return -1.0 if reverse else 101.0

                        elif col in [
                            "turn_count",
                            "token_consumption",
                            "end_to_end_latency",
                        ]:
                            if val == "N/A":
                                return -1.0 if reverse else 1e12
                            try:
                                return float(val)
                            except ValueError:
                                return -1.0 if reverse else 1e12

                        # String columns (product, requester, id, date)
                        if val == "N/A":
                            return "" if reverse else "\xff\xff\xff\xff"
                        return str(val)

                    summaries.sort(key=get_sort_key, reverse=reverse)

                    # Extract unique values for filters from ALL summaries
                    all_summaries = []
                    if s.eval_summaries:
                        try:
                            all_summaries = json.loads(s.eval_summaries)
                        except Exception:
                            all_summaries = []

                    products = sorted(
                        list(
                            set(
                                x["product"]
                                for x in all_summaries
                                if x["product"] != "N/A"
                            )
                        )
                    )
                    requesters = sorted(
                        list(
                            set(
                                x.get("requester", "N/A")
                                for x in all_summaries
                                if x.get("requester", "N/A") != "N/A"
                            )
                        )
                    )
                    eval_ids = sorted([x["id"] for x in all_summaries])

                    # Apply filters
                    if state.eval_id_filter:
                        summaries = [
                            x
                            for x in summaries
                            if x["id"] == state.eval_id_filter
                        ]
                    if state.product_filter:
                        summaries = [
                            x
                            for x in summaries
                            if x["product"] == state.product_filter
                        ]
                    if state.requester_filter:
                        summaries = [
                            x
                            for x in summaries
                            if x.get("requester", "N/A")
                            == state.requester_filter
                        ]

                    # Render filters UI
                    with me.box(
                        style=me.Style(
                            display="flex",
                            flex_direction="row",
                            gap="24px",
                            margin=me.Margin(top="16px", bottom="24px"),
                            padding=me.Padding.all("16px"),
                            background="#ffffff",
                            border_radius="12px",
                            box_shadow=(
                                "0 1px 3px 0 rgba(0, 0, 0, 0.1), "
                                "0 1px 2px -1px rgba(0, 0, 0, 0.1)"
                            ),
                            align_items="center",
                            border=me.Border.all(
                                me.BorderSide(
                                    width="1px",
                                    color="#e2e8f0",
                                    style="solid",
                                )
                            ),
                        )
                    ):
                        def toggle_eval_id_dropdown(e: me.ClickEvent):
                            st = me.state(State)
                            if st.open_dropdown == "eval_id":
                                st.open_dropdown = ""
                            else:
                                st.open_dropdown = "eval_id"

                        def make_eval_id_handler(val):
                            def handler(e: me.ClickEvent):
                                st = me.state(State)
                                st.eval_id_filter = val
                                st.open_dropdown = ""

                            handler.__name__ = f"click_eval_id_{val}"
                            return handler

                        with me.box(
                            style=me.Style(
                                position="relative",
                                width="200px",
                            )
                        ):
                            # The Box acting as Dropdown Trigger
                            with me.box(
                                style=me.Style(
                                    background="#ffffff",
                                    border=me.Border.all(
                                        me.BorderSide(
                                            width="1px",
                                            color="#e2e8f0",
                                        )
                                    ),
                                    border_radius="4px",
                                    padding=me.Padding.all("8px"),
                                    cursor="pointer",
                                ),
                                on_click=toggle_eval_id_dropdown,
                            ):
                                me.text(
                                    state.eval_id_filter
                                    if state.eval_id_filter
                                    else "Select Eval ID",
                                    style=me.Style(
                                        color="#1f2937"
                                    ),
                                )

                            # The Popup List
                            if state.open_dropdown == "eval_id":
                                with me.box(
                                    style=me.Style(
                                        position="absolute",
                                        top="100%",
                                        left="0",
                                        z_index=10,
                                        background="#ffffff",
                                        border=me.Border.all(
                                            me.BorderSide(
                                                width="1px",
                                                color="#e2e8f0",
                                            )
                                        ),
                                        border_radius="4px",
                                        width="100%",
                                        max_height="200px",
                                        overflow_y="auto",
                                    )
                                ):
                                    # All option
                                    with me.box(
                                        style=me.Style(
                                            padding=me.Padding.all("8px"),
                                            cursor="pointer",
                                        ),
                                        on_click=make_eval_id_handler(""),
                                    ):
                                        me.text(
                                            "All",
                                            style=me.Style(
                                                color="#1f2937"
                                            ),
                                        )

                                    for d in eval_ids:
                                        with me.box(
                                            style=me.Style(
                                                padding=me.Padding.all("8px"),
                                                cursor="pointer",
                                            ),
                                            on_click=make_eval_id_handler(d),
                                        ):
                                            me.text(
                                                d,
                                                style=me.Style(
                                                    color="#1f2937"
                                                ),
                                            )

                        # Product Filter with Floating Autocomplete
                        def toggle_product_dropdown(e: me.ClickEvent):
                            st = me.state(State)
                            if st.open_dropdown == "product":
                                st.open_dropdown = ""
                            else:
                                st.open_dropdown = "product"

                        def make_prod_dropdown_handler(val):
                            def handler(e: me.ClickEvent):
                                st = me.state(State)
                                st.product_filter = val
                                st.open_dropdown = ""

                            handler.__name__ = f"click_prod_dd_{val}"
                            return handler

                        mk_prod_dd = make_prod_dropdown_handler

                        with me.box(
                            style=me.Style(
                                position="relative",
                                width="200px",
                            )
                        ):
                            # The Box acting as Dropdown Trigger
                            with me.box(
                                style=me.Style(
                                    background="#ffffff",
                                    border=me.Border.all(
                                        me.BorderSide(
                                            width="1px",
                                            color="#e2e8f0",
                                        )
                                    ),
                                    border_radius="4px",
                                    padding=me.Padding.all("8px"),
                                    cursor="pointer",
                                ),
                                on_click=toggle_product_dropdown,
                            ):
                                me.text(
                                    state.product_filter
                                    if state.product_filter
                                    else "Filter by Product",
                                    style=me.Style(
                                        color="#1f2937"
                                    ),
                                )

                            # The Popup List
                            if state.open_dropdown == "product":
                                with me.box(
                                    style=me.Style(
                                        position="absolute",
                                        top="100%",
                                        left="0",
                                        z_index=10,
                                        background="#ffffff",
                                        border=me.Border.all(
                                            me.BorderSide(
                                                width="1px",
                                                color="#e2e8f0",
                                            )
                                        ),
                                        border_radius="4px",
                                        width="100%",
                                        max_height="200px",
                                        overflow_y="auto",
                                    )
                                ):
                                    # All option
                                    with me.box(
                                        style=me.Style(
                                            padding=me.Padding.all("8px"),
                                            cursor="pointer",
                                        ),
                                        on_click=mk_prod_dd(""),
                                    ):
                                        me.text(
                                            "All",
                                            style=me.Style(
                                                color="#1f2937"
                                            ),
                                        )

                                    for p in products:
                                        with me.box(
                                            style=me.Style(
                                                padding=me.Padding.all("8px"),
                                                cursor="pointer",
                                            ),
                                            on_click=mk_prod_dd(p),
                                        ):
                                            me.text(
                                                p,
                                                style=me.Style(
                                                    color="#1f2937"
                                                ),
                                            )

                        # Requester Filter with Floating Autocomplete
                        def toggle_requester_dropdown(e: me.ClickEvent):
                            st = me.state(State)
                            if st.open_dropdown == "requester":
                                st.open_dropdown = ""
                            else:
                                st.open_dropdown = "requester"

                        def make_req_dropdown_handler(val):
                            def handler(e: me.ClickEvent):
                                st = me.state(State)
                                st.requester_filter = val
                                st.open_dropdown = ""

                            handler.__name__ = f"click_req_dd_{val}"
                            return handler

                        mk_req_dd = make_req_dropdown_handler

                        with me.box(
                            style=me.Style(
                                position="relative",
                                width="200px",
                            )
                        ):
                            # The Box acting as Dropdown Trigger
                            with me.box(
                                style=me.Style(
                                    background="#ffffff",
                                    border=me.Border.all(
                                        me.BorderSide(
                                            width="1px",
                                            color="#e2e8f0",
                                        )
                                    ),
                                    border_radius="4px",
                                    padding=me.Padding.all("8px"),
                                    cursor="pointer",
                                ),
                                on_click=toggle_requester_dropdown,
                            ):
                                me.text(
                                    state.requester_filter
                                    if state.requester_filter
                                    else "Filter by Requester",
                                    style=me.Style(
                                        color="#1f2937"
                                    ),
                                )

                            # The Popup List
                            if state.open_dropdown == "requester":
                                with me.box(
                                    style=me.Style(
                                        position="absolute",
                                        top="100%",
                                        left="0",
                                        z_index=10,
                                        background="#ffffff",
                                        border=me.Border.all(
                                            me.BorderSide(
                                                width="1px",
                                                color="#e2e8f0",
                                            )
                                        ),
                                        border_radius="4px",
                                        width="100%",
                                        max_height="200px",
                                        overflow_y="auto",
                                    )
                                ):
                                    # All option
                                    with me.box(
                                        style=me.Style(
                                            padding=me.Padding.all("8px"),
                                            cursor="pointer",
                                        ),
                                        on_click=mk_req_dd(""),
                                    ):
                                        me.text(
                                            "All",
                                            style=me.Style(
                                                color="#1f2937"
                                            ),
                                        )

                                    for r in requesters:
                                        with me.box(
                                            style=me.Style(
                                                padding=me.Padding.all("8px"),
                                                cursor="pointer",
                                            ),
                                            on_click=mk_req_dd(r),
                                        ):
                                            me.text(
                                                r,
                                                style=me.Style(
                                                    color="#1f2937"
                                                ),
                                            )

                    def on_sort_click(col_name):
                        s = me.state(State)
                        if s.sort_column == col_name:
                            s.sort_descending = not s.sort_descending
                        else:
                            s.sort_column = col_name
                            s.sort_descending = True

                    def click_id(e):
                        on_sort_click("id")

                    def click_date(e):
                        on_sort_click("date")

                    def click_product(e):
                        on_sort_click("product")

                    def click_requester(e):
                        on_sort_click("requester")

                    def click_traj(e):
                        on_sort_click("trajectory_matcher")

                    def click_turns(e):
                        on_sort_click("turn_count")

                    def click_exec(e):
                        on_sort_click("executable")

                    def click_tokens(e):
                        on_sort_click("token_consumption")

                    def click_latency(e):
                        on_sort_click("end_to_end_latency")

                    sort_handlers = {
                        "id": click_id,
                        "date": click_date,
                        "product": click_product,
                        "requester": click_requester,
                        "trajectory_matcher": click_traj,
                        "turn_count": click_turns,
                        "executable": click_exec,
                        "token_consumption": click_tokens,
                        "end_to_end_latency": click_latency,
                    }

                    def render_header_cell(h_label, h_col, h_width):
                        with me.box(
                            style=me.Style(
                                display="table-cell",
                                padding=me.Padding.symmetric(
                                    vertical="12px", horizontal="16px"
                                ),
                                text_align="center",
                                border=me.Border.all(
                                    me.BorderSide(
                                        width="1px",
                                        color="#e2e8f0",
                                        style="solid",
                                    )
                                ),
                                cursor="pointer",
                                width=h_width,
                                white_space="nowrap" if h_width else None,
                                background="#f8fafc",
                            ),
                            on_click=sort_handlers[h_col],
                        ):
                            with me.box(
                                style=me.Style(
                                    display="flex",
                                    align_items="center",
                                    justify_content="center",
                                    color="#475569",
                                )
                            ):
                                me.text(h_label)
                                s = me.state(State)
                                arrow = (
                                    " ↓" if s.sort_descending else " ↑"
                                )
                                arrow_str = (
                                    arrow
                                    if s.sort_column == h_col
                                    else ""
                                )
                                me.text(
                                    arrow_str,
                                    style=me.Style(
                                        font_weight="bold",
                                        color="#0284c7",
                                        font_size="14px",
                                        margin=me.Margin(left="4px"),
                                    ),
                                )

                    with me.box(
                        style=me.Style(
                            max_height="600px",
                            overflow_y="auto",
                            margin=me.Margin(top="16px"),
                            display="table",
                            width="100%",
                            border=me.Border.all(
                                me.BorderSide(
                                    width="1px",
                                    color="#e5e7eb",
                                    style="solid",
                                )
                            ),
                            border_radius="8px",
                            background="#ffffff",
                        )
                    ):
                        # Header row
                        with me.box(
                            style=me.Style(
                                display="table-row",
                                background="#f8fafc",
                                font_weight="bold",
                                color="#475569",
                                font_size="12px",
                                text_transform="uppercase",
                                letter_spacing="0.05em",
                            )
                        ):
                            headers = [
                                ("Eval ID", "id", "36ch"),
                                ("Date", "date", "24ch"),
                                ("Product", "product", None),
                                ("Requester", "requester", None),
                                (
                                    "Trajectory Matcher",
                                    "trajectory_matcher",
                                    "18ch",
                                ),
                                ("Turn Count", "turn_count", "12ch"),
                                ("Executable", "executable", "12ch"),
                                (
                                    "Token Consumption",
                                    "token_consumption",
                                    "16ch",
                                ),
                                (
                                    "End-to-End Latency",
                                    "end_to_end_latency",
                                    "20ch",
                                ),
                            ]
                            for label, col, width in headers:
                                render_header_cell(label, col, width)

                        # Data rows
                        for idx, item in enumerate(summaries):
                            d = item["id"]
                            date_val = item.get("date", "N/A")
                            prod = item["product"]
                            req_val = item.get("requester", "N/A")
                            traj = item.get("trajectory_matcher", "N/A")
                            turns = item.get("turn_count", "N/A")
                            exec_val = item.get("executable", "N/A")
                            tokens = item.get("token_consumption", "N/A")
                            latency = item.get("end_to_end_latency", "N/A")

                            bg_color = (
                                "#ffffff"
                                if idx % 2 == 0
                                else "#f8fafc"
                            )

                            def make_on_click(dir_name):
                                def on_click(e: me.ClickEvent):
                                    s = me.state(State)
                                    s.selected_directory = dir_name
                                return on_click

                            with me.box(
                                style=me.Style(
                                    display="table-row",
                                    background=bg_color,
                                )
                            ):
                                # Eval ID as a link/button
                                with me.box(
                                    style=me.Style(
                                        display="table-cell",
                                        padding=me.Padding.symmetric(
                                            vertical="10px", horizontal="16px"
                                        ),
                                        text_align="center",
                                        border=me.Border.all(
                                            me.BorderSide(
                                                width="1px",
                                                color="#e2e8f0",
                                                style="solid",
                                            )
                                        ),
                                        width="36ch",
                                        white_space="nowrap",
                                    )
                                ):
                                    me.button(
                                        d,
                                        on_click=make_on_click(d),
                                        style=me.Style(
                                            text_align="center",
                                            background="transparent",
                                            color="#0284c7",
                                            font_family="monospace",
                                            font_size="14px",
                                            padding=me.Padding.all("0px"),
                                            margin=me.Margin.all("0px"),
                                            border=me.Border.all(
                                                me.BorderSide(width="0px")
                                            ),
                                            font_weight="500",
                                            width="100%",
                                        ),
                                    )
                                with me.box(
                                    style=me.Style(
                                        display="table-cell",
                                        padding=me.Padding.symmetric(
                                            vertical="10px", horizontal="16px"
                                        ),
                                        text_align="center",
                                        border=me.Border.all(
                                            me.BorderSide(
                                                width="1px",
                                                color="#e2e8f0",
                                                style="solid",
                                            )
                                        ),
                                        width="24ch",
                                        white_space="nowrap",
                                    )
                                ):
                                    me.text(
                                        date_val,
                                        style=me.Style(
                                            color="#334155",
                                            font_family="monospace",
                                        ),
                                    )
                                with me.box(
                                    style=me.Style(
                                        display="table-cell",
                                        padding=me.Padding.symmetric(
                                            vertical="10px", horizontal="16px"
                                        ),
                                        text_align="center",
                                        border=me.Border.all(
                                            me.BorderSide(
                                                width="1px",
                                                color="#e2e8f0",
                                                style="solid",
                                            )
                                        ),
                                    )
                                ):
                                    me.text(
                                        prod,
                                        style=me.Style(
                                            color="#334155"
                                        ),
                                    )

                                with me.box(
                                    style=me.Style(
                                        display="table-cell",
                                        padding=me.Padding.symmetric(
                                            vertical="10px", horizontal="16px"
                                        ),
                                        text_align="center",
                                        border=me.Border.all(
                                            me.BorderSide(
                                                width="1px",
                                                color="#e2e8f0",
                                                style="solid",
                                            )
                                        ),
                                    )
                                ):
                                    me.text(
                                        req_val,
                                        style=me.Style(
                                            color="#334155"
                                        ),
                                    )

                                with me.box(
                                    style=me.Style(
                                        display="table-cell",
                                        padding=me.Padding.symmetric(
                                            vertical="10px", horizontal="16px"
                                        ),
                                        text_align="center",
                                        border=me.Border.all(
                                            me.BorderSide(
                                                width="1px",
                                                color="#e2e8f0",
                                                style="solid",
                                            )
                                        ),
                                        width="18ch",
                                        white_space="nowrap",
                                    )
                                ):
                                    me.text(
                                        traj,
                                        style=me.Style(
                                            color=get_color_for_pct(traj)
                                        ),
                                    )

                                with me.box(
                                    style=me.Style(
                                        display="table-cell",
                                        padding=me.Padding.symmetric(
                                            vertical="10px", horizontal="16px"
                                        ),
                                        text_align="center",
                                        border=me.Border.all(
                                            me.BorderSide(
                                                width="1px",
                                                color="#e2e8f0",
                                                style="solid",
                                            )
                                        ),
                                    )
                                ):
                                    me.text(
                                        turns,
                                        style=me.Style(
                                            color="#334155"
                                        ),
                                    )

                                with me.box(
                                    style=me.Style(
                                        display="table-cell",
                                        padding=me.Padding.symmetric(
                                            vertical="10px", horizontal="16px"
                                        ),
                                        text_align="center",
                                        border=me.Border.all(
                                            me.BorderSide(
                                                width="1px",
                                                color="#e2e8f0",
                                                style="solid",
                                            )
                                        ),
                                    )
                                ):
                                    me.text(
                                        exec_val,
                                        style=me.Style(
                                            color=get_color_for_pct(exec_val)
                                        ),
                                    )

                                with me.box(
                                    style=me.Style(
                                        display="table-cell",
                                        padding=me.Padding.symmetric(
                                            vertical="10px", horizontal="16px"
                                        ),
                                        text_align="center",
                                        border=me.Border.all(
                                            me.BorderSide(
                                                width="1px",
                                                color="#e2e8f0",
                                                style="solid",
                                            )
                                        ),
                                    )
                                ):
                                    me.text(
                                        tokens,
                                        style=me.Style(
                                            color="#334155"
                                        ),
                                    )

                                with me.box(
                                    style=me.Style(
                                        display="table-cell",
                                        padding=me.Padding.symmetric(
                                            vertical="10px", horizontal="16px"
                                        ),
                                        text_align="center",
                                        border=me.Border.all(
                                            me.BorderSide(
                                                width="1px",
                                                color="#e2e8f0",
                                                style="solid",
                                            )
                                        ),
                                    )
                                ):
                                    me.text(
                                        latency,
                                        style=me.Style(
                                            color="#334155"
                                        ),
                                    )


if __name__ == "__main__":
    me.run(app)
