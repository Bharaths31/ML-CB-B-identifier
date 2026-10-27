import { createBrowserRouter, RouterProvider } from "react-router-dom";
import UserPage from "./pages/UserPage.jsx";
import DevPage from "./pages/DevPage.jsx";
import ExpertPage from "./pages/ExpertPage.jsx";

const router = createBrowserRouter([
  {
    path: "/",
    element: <UserPage />,
  },
  {
    path: "/dev",
    element: <DevPage />,
  },
  {
    path: "/expert",
    element: <ExpertPage />,
  },
]);

export default function AppRouter() {
  return <RouterProvider router={router} />;
}
