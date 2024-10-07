import streamlit as st
from create_db import register_user, login_user
import quick_ml
from quick_ml import get_cleaned_data
import re

def validate_username(username):
    if len(username) < 5 or not username.isalnum():
        return False
    return True

def validate_password(password):
    if (len(password) < 8 or 
        not re.search(r"[A-Z]", password) or
        not re.search(r"[a-z]", password) or
        not re.search(r"\d", password) or
        not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password)):
        return False
    return True

def main():
    st.title("Quick ML")

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'username' not in st.session_state:
        st.session_state.username = None

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
                if not validate_username(new_user):
                    st.error("Username must be at least 5 characters long and contain only alphanumeric characters.")
                elif not validate_password(new_password):
                    st.error("Password must be at least 8 characters long, include an uppercase letter, lowercase letter, digit, and special character.")
                else:
                    if register_user(new_user, new_password):
                        st.success("Account created successfully. Please login.")
                    else:
                        st.error("Username already exists. Try a different one.")

    else:
        st.sidebar.success(f"Logged in as {st.session_state.username}")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.username = None
            st.rerun()
        else:
            quick_ml.main(st.session_state.user_id)

if __name__ == '__main__':
    main()