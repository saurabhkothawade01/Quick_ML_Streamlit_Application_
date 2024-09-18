import streamlit as st
from create_db import register_user, login_user
import quick_ml
from quick_ml import get_cleaned_data

def main():
    st.title("Quick ML")

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = None

    if not st.session_state.logged_in:
        menu = ["Login", "Register"]
        choice = st.sidebar.selectbox("Menu", menu)

        if choice == "Login":
            st.subheader("Login Section")

            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                user = login_user(username, password)
                if user:
                    st.session_state.user_id = user[0] 
                    st.session_state.username = user[1]
                    st.session_state.logged_in = True
                    st.success(f"Welcome {st.session_state.username}")
                    st.rerun()
                else:
                    st.error("Invalid Username or Password")

        elif choice == "Register":
            st.subheader("Create New Account")

            new_user = st.text_input("Username")
            new_password = st.text_input("Password", type="password")

            if st.button("Register"):
                if register_user(new_user, new_password):
                    st.success("Account created successfully. Please login.")
                else:
                    st.error("Username already exists. Try a different one.")

    else:
        st.sidebar.success(f"Logged in as {st.session_state.username}")
        
        if st.sidebar.button("Download cleaned data"):
            cleaned_data = get_cleaned_data()
            if cleaned_data is not None:
                try:
                    st.sidebar.download_button(
                        label="Download CSV",
                        data=cleaned_data.to_csv(index=False).encode('utf-8'),
                        file_name="cleaned_data.csv",
                        mime="text/csv"
                    )
                    st.sidebar.success("Download ready. Click the 'Download CSV' button to save the file.")
                except Exception as e:
                    st.sidebar.error(f"Error preparing download: {e}")
            else:
                st.sidebar.warning("No cleaned data available. Please analyze data first.")

        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.username = None
            st.session_state.cleaned_data = None
            st.rerun()
        else:
            quick_ml.main(st.session_state.user_id)

if __name__ == '__main__':
    main()