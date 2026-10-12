import { createBrowserRouter, RouterProvider } from "react-router-dom";
import UserPage from "./pages/UserPage.jsx";
import DevPage from "./pages/DevPage.jsx";

const router = createBrowserRouter([
  {
    path: "/",
    element: <UserPage />,
  },
  {
    path: "/dev",
    element: <DevPage />,
  },
]);

export default function AppRouter() {
  return <RouterProvider router={router} />;
}
